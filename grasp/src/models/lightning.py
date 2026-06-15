from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from scipy.spatial.transform import Rotation
from torch import Tensor

from src.core.config import ExperimentConfig
from src.core.visualize import check_collision, scene_to_wandb_3d
from src.data.util import GraspData, denormalize_translation
from src.models.flow import sample, sample_location_and_conditional_flow, rotmat_to_rotvec
from src.models.util import get_grasp_from_batch
from src.models.velocity_mlp import VelocityNetwork
from src.models.pcgrad import PCGrad
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
import numpy as np


class Lightning(pl.LightningModule):
    """Flow Matching model combining SO3 and R3 manifold learning with synchronized time sampling."""

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.model = VelocityNetwork(self.config)

        # TODO use config
        self.save_hyperparameters()
        self.all_collision = []
        self.all_graspable = []
        self.all_wasserstein_so3 = []
        self.all_wasserstein_r3 = []
        self.steps = config.training.steps
        self.all_steps_success = []

        self.translation_norm_params = 0
        self.all_success = []
        self.method = config.training.method
        self.step_collec = [1,2,3,4,5,6,7]

    def lsd_loss(
            self,
            so3_inputs: Tensor,
            r3_inputs: Tensor,
            sdf_inputs: Tensor,
            sdf_path: Tuple[str],
            # dataset_mesh_scale: float,
            normalization_scale: float,
            prefix: str = "train",
    ):
        so3_inputs = self.model.duplicate_to_batch_size(
            so3_inputs,
            self.config.data.batch_size,
            self.config.training.duplicate_ratio,
        )
        r3_inputs = self.model.duplicate_to_batch_size(
            r3_inputs, self.config.data.batch_size, self.config.training.duplicate_ratio
        )
        t = torch.rand(r3_inputs.size(0), device=so3_inputs.device)
        r = torch.rand(r3_inputs.size(0), device=so3_inputs.device) * t

        # SO3 computation - already in [batch, 3, 3] format
        x0_so3 = torch.tensor(
            Rotation.random(r3_inputs.size(0)).as_matrix(), device=so3_inputs.device
        )  # Shape: [batch, 3, 3]

        # Sample location and flow for SO
        xr_so3, vr_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, r)
        # Both xt_so3 and ut_so3 are [batch, 3, 3]

        t_expanded = t.unsqueeze(-1)  # [batch, 1]
        r_expanded = r.unsqueeze(-1)

        # x0_r3
        x0_r3 = torch.randn_like(r3_inputs)

        # Get predicted flow for R3
        xt_r3 = (1 - (1 - self.config.model.sigma_min) * r_expanded) * x0_r3 + r_expanded * r3_inputs
        # r3 speed
        vr_r3 = r3_inputs - (1 - self.config.model.sigma_min) * x0_r3

        def temp_xrt(xr_so3, xr_r3, r, t):
            return self.xrt(xr_so3, xr_r3, r, t, sdf_inputs, normalization_scale, sdf_path)

        xst, dvdt = torch.func.jvp(
            lambda _t: temp_xrt(xr_so3, xt_r3, r_expanded, _t),
            primals=(t_expanded,),
            tangents=(torch.ones_like(t_expanded),),
        )
        xst_so3 = xst[0]
        xst_so3 = xst_so3.detach()
        xst_r3 = xst[1]
        xst_r3 = xst_r3.detach()
        dvdt_so3 = dvdt[0]
        dvdt_r3 = dvdt[1]
            # print('xst_so3:', xst_so3.shape)
            # print('xst_r3:', xst_r3.shape)
            # print('sdf_inputs:', sdf_inputs.shape)
            # print('xst_r3:', xst_r3.shape)
            # exit(0)
        with torch.no_grad():
            ref_so3, ref_r3 = self.model.forward(
                xst_so3, xst_r3, sdf_inputs, t_expanded, r_expanded, normalization_scale, sdf_path
            )

        diff_so3 = ref_so3 - dvdt_so3

        rie = torch.transpose(xst_so3, dim0=-2, dim1=-1) @ diff_so3
        norm = -torch.diagonal(rie @ rie, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss = torch.mean(norm, dim=-1)

        # Compute noisy sample and optimal flow for R3
        r3_loss = F.mse_loss(ref_r3, dvdt_r3)

        # Works better in this setup but we can change later
        total_loss = (so3_loss + 2 * r3_loss)

        # fm_loss
        with torch.no_grad():
            xt_so3, vt_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, t)
            xt_r3 = (1 - (1 - self.config.model.sigma_min) * t_expanded
                    ) * x0_r3 + t_expanded * r3_inputs
            vt_r3 = r3_inputs - (1 - self.config.model.sigma_min) * x0_r3
        if vt_so3.isnan().any():
            nans = vt_so3.isnan()
            while nans.ndim > 1:
                nans = nans.any(dim=-1)
            vt_so3 = vt_so3[~nans]
            xt_so3 = xt_so3[~nans]
            t = t[~nans]
            xt_r3 = xt_r3[~nans]
            vt_r3 = vt_r3[~nans]
        fm_so3, fm_r3 = self.model.forward(
            xt_so3, xt_r3, sdf_inputs, t.unsqueeze(-1), t.unsqueeze(-1), normalization_scale, sdf_path
        )

        rie = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (fm_so3 - vt_so3)
        norm = -torch.diagonal(rie @ rie, dim1=-2, dim2=-1).sum(dim=-1) / 2

        so3_loss_fm = torch.mean(norm, dim=-1)

        r3_loss_fm = F.mse_loss(fm_r3, vt_r3)

        fm_loss = so3_loss_fm + 2 * r3_loss_fm

        # FM + LSD
        total_loss = total_loss + fm_loss

        loss_dict = {
            f"{prefix}/so3_loss": so3_loss,
            f"{prefix}/r3_loss": r3_loss,
            f"{prefix}/loss": total_loss,
            f"{prefix}/fm_loss": fm_loss,
        }
        total_loss = total_loss if torch.isfinite(total_loss) else None

        return total_loss, loss_dict

    def so3_exp(self, x0, x1):
        vec_manifold = SpecialOrthogonal(n=3, point_type="vector")

        # Convert rotations to axis-angle representation and compute log map
        rot_x0 = rotmat_to_rotvec(x0)
        rot_x1 = rotmat_to_rotvec(x1)

        log_x1 = vec_manifold.log_not_from_identity(rot_x1, rot_x0)

        # Compute interpolated rotation at time t
        xt = vec_manifold.exp_not_from_identity(log_x1, rot_x0)
        xt = vec_manifold.matrix_from_rotation_vector(xt)
        return xt

    def xrt(self, xr_so3, xr_r3, r, t, sdf_inputs, normalization_scale, sdf_path):
        vt_so3, vt_r3 = self.model.forward(
            xr_so3, xr_r3, sdf_inputs, t, r, normalization_scale, sdf_path
        )
        # print('t:', t.shape)
        # print('r:', r.shape)
        # print('vt_so3:', vt_so3.shape)
        # print('vt_so3:', vt_r3.shape)
        # exit(0)
        xrrt_so3 = self.so3_exp(xr_so3, (t.unsqueeze(-1) - r.unsqueeze(-1)) * vt_so3)
        xrrt_r3 = xr_r3 + (t - r) * vt_r3
        return xrrt_so3, xrrt_r3

    def compute_loss_pc(
            self,
            so3_inputs: Tensor,
            r3_inputs: Tensor,
            sdf_inputs: Tensor,
            sdf_path: Tuple[str],
            # dataset_mesh_scale: float,
            normalization_scale: float,
            prefix: str = "train",
    ) -> Tuple[Tensor, Tensor,Dict[str, Tensor]]:
        """Compute combined loss for both manifolds with synchronized time sampling.

        Args:
            so3_inputs: Target SO3 matrices [batch, 3, 3]
            r3_inputs: Target R3 points [batch, 3]
            prefix: Prefix for logging metrics

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Sample synchronized time points for both manifolds
        with torch.no_grad():
            so3_inputs = self.model.duplicate_to_batch_size(
                so3_inputs,
                self.config.data.batch_size,
                self.config.training.duplicate_ratio,
            )
            r3_inputs = self.model.duplicate_to_batch_size(
                r3_inputs, self.config.data.batch_size, self.config.training.duplicate_ratio
            )
            t = torch.rand(r3_inputs.size(0), device=so3_inputs.device)
            r = torch.rand(r3_inputs.size(0), device=so3_inputs.device) * t

            # SO3 computation - already in [batch, 3, 3] format
            x0_so3 = torch.tensor(
                Rotation.random(r3_inputs.size(0)).as_matrix(), device=so3_inputs.device
            )  # Shape: [batch, 3, 3]

            # Sample location and flow for SO
            xt_so3, vt_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, t)
            # Both xt_so3 and ut_so3 are [batch, 3, 3]

            t_expanded = t.unsqueeze(-1)  # [batch, 1]
            r_expanded = r.unsqueeze(-1)

            # x0_r3
            noise = torch.randn_like(r3_inputs)

            # Get predicted flow for R3
            x_t_r3 = (
                             1 - (1 - self.config.model.sigma_min) * t_expanded
                     ) * noise + t_expanded * r3_inputs
            # r3 speed
            vt_r3 = r3_inputs - (1 - self.config.model.sigma_min) * noise

        def temp_forward(temp_xt_so3, temp_xt_r3, temp_t, temp_r):
            vt_so3_sita, vt_r3_sita = self.model.forward(
                temp_xt_so3, temp_xt_r3, sdf_inputs, temp_t, temp_r, normalization_scale, sdf_path
            )
            return vt_so3_sita, vt_r3_sita

        # vt_so3_imf, vt_r3_imf = self.model.forward(xt_so3, x_t_r3, sdf_inputs, t_expanded, r_expanded, normalization_scale, sdf_path)

        # average speed
        se3_u, se3_dudt = torch.func.jvp(
            temp_forward,
            (xt_so3, x_t_r3, t_expanded, r_expanded),
            (vt_so3, vt_r3, torch.ones_like(t_expanded), torch.zeros_like(r_expanded)),
        )
        ut_so3 = se3_u[0]
        ut_r3 = se3_u[1]
        dudt_so3 = se3_dudt[0]
        dudt_r3 = se3_dudt[1]
        # exit(0)

        rie = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (ut_so3 - vt_so3)
        norm_so3_loss_1 = -torch.diagonal(rie @ rie, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss_1 = torch.mean(norm_so3_loss_1, dim=-1)

        rie_0 = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ ut_so3
        rie_1 = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ ((t_expanded.unsqueeze(-1) - r_expanded.unsqueeze(-1)) * 2 * dudt_so3.detach())
        norm_so3_loss_2 = -torch.diagonal(rie_0 @ rie_1, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss_2 = torch.mean(norm_so3_loss_2, dim=-1)


        # Compute noisy sample and optimal flow for R3
        # vt_r3 = r3_inputs - (1 - self.config.model.sigma_min) * noise
        r3_loss_1 = F.mse_loss(ut_r3, vt_r3)
        r3_loss_2 = torch.sum(ut_r3 * (t_expanded - r_expanded) * 2 * dudt_r3.detach(), dim=-1).mean()

        # Works better in this setup but we can change later
        total_loss_1 = (
                self.config.training.so3_loss_weight * so3_loss_1
                + self.config.training.r3_loss_weight * r3_loss_1
        )
        total_loss_2 =  (
                self.config.training.so3_loss_weight * so3_loss_2
                + self.config.training.r3_loss_weight * r3_loss_2
        )
        # if total_loss.isnan().any():
        #     nans = total_loss.isnan().any(dim=(-1, -2))
        #     total_loss = total_loss[~nans]

        loss_dict = {
            f"{prefix}/so3_loss": so3_loss_1 + so3_loss_2,
            f"{prefix}/r3_loss": r3_loss_1 + r3_loss_2,
            f"{prefix}/loss": total_loss_1 + total_loss_2,
        }

        return total_loss_1, total_loss_2, loss_dict

    def compute_loss(
        self,
        so3_inputs: Tensor,
        r3_inputs: Tensor,
        sdf_inputs: Tensor,
        sdf_path: Tuple[str],
        # dataset_mesh_scale: float,
        normalization_scale: float,
        prefix: str = "train",
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute combined loss for both manifolds with synchronized time sampling.

        Args:
            so3_inputs: Target SO3 matrices [batch, 3, 3]
            r3_inputs: Target R3 points [batch, 3]
            prefix: Prefix for logging metrics

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Sample synchronized time points for both manifolds
        with torch.no_grad():

            so3_inputs = self.model.duplicate_to_batch_size(
                so3_inputs,
                self.config.data.batch_size,
                self.config.training.duplicate_ratio,
            )
            r3_inputs = self.model.duplicate_to_batch_size(
                r3_inputs, self.config.data.batch_size, self.config.training.duplicate_ratio
            )
            t = torch.rand(r3_inputs.size(0), device=so3_inputs.device)
            r = torch.rand(r3_inputs.size(0), device=so3_inputs.device) * t

            # SO3 computation - already in [batch, 3, 3] format
            x0_so3 = torch.tensor(
                Rotation.random(r3_inputs.size(0)).as_matrix(), device=so3_inputs.device
            )  # Shape: [batch, 3, 3]

            # Sample location and flow for SO
            xt_so3, vt_so3 = sample_location_and_conditional_flow(x0_so3, so3_inputs, t)
            # Both xt_so3 and ut_so3 are [batch, 3, 3]

            t_expanded = t.unsqueeze(-1)  # [batch, 1]
            r_expanded = r.unsqueeze(-1)

            # x0_r3
            noise = torch.randn_like(r3_inputs)

            # Get predicted flow for R3
            x_t_r3 = (
                1 - (1 - self.config.model.sigma_min) * t_expanded
            ) * noise + t_expanded * r3_inputs
            # r3 speed
            vt_r3 = r3_inputs - (1 - self.config.model.sigma_min) * noise

        def temp_forward(temp_xt_so3, temp_xt_r3, temp_t, temp_r):
            vt_so3_sita, vt_r3_sita = self.model.forward(
                temp_xt_so3, temp_xt_r3, sdf_inputs, temp_t, temp_r, normalization_scale, sdf_path
            )
            return vt_so3_sita, vt_r3_sita

        # vtt_so3, vtt_r3 = self.model.forward(
        #     xt_so3, x_t_r3, sdf_inputs, t_expanded, r_expanded, normalization_scale, sdf_path
        # )

        # average speed
        se3_u, se3_dudt = torch.func.jvp(
            temp_forward,
            (xt_so3, x_t_r3, t_expanded, r_expanded),
            (vt_so3, vt_r3, torch.ones_like(t_expanded), torch.zeros_like(r_expanded),),
        )
        ut_so3 = se3_u[0]
        ut_r3 = se3_u[1]
        dudt_so3 = se3_dudt[0]
        dudt_r3 = se3_dudt[1]
        with torch.no_grad():
            u_so3_tgt = vt_so3 - (t_expanded.unsqueeze(-1) - r_expanded.unsqueeze(-1)) * dudt_so3.detach()
            u_r3_tgt = vt_r3 - (t_expanded - r_expanded) * dudt_r3.detach()
        # vt_so3 is now directly [batch, 3, 3]

        # Compute SO3 loss using Riemannian metric
        rie = torch.transpose(xt_so3, dim0=-2, dim1=-1) @ (ut_so3 - u_so3_tgt)
        norm = -torch.diagonal(rie @ rie, dim1=-2, dim2=-1).sum(dim=-1) / 2
        so3_loss = torch.mean(norm, dim=-1)

        # Compute noisy sample and optimal flow for R3
        # vt_r3 = r3_inputs - (1 - self.config.model.sigma_min) * noise
        r3_loss = F.mse_loss(ut_r3, u_r3_tgt)

        # Works better in this setup but we can change later
        total_loss = (
            self.config.training.so3_loss_weight * so3_loss
            + self.config.training.r3_loss_weight * r3_loss
        )

        loss_dict = {
            f"{prefix}/so3_loss": so3_loss,
            f"{prefix}/r3_loss": r3_loss,
            f"{prefix}/loss": total_loss,
        }

        return total_loss, loss_dict

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        grasp_data = batch
        if self.method == 'mf':
            loss, log_dict = self.compute_loss(
                grasp_data.rotation,
                grasp_data.translation,
                grasp_data.sdf,
                grasp_data.mesh_path,
                grasp_data.normalization_scale,
                "train",
            )
            self.log_dict(
                log_dict,
                prog_bar=True,
                batch_size=self.config.data.batch_size,
            )

            return loss
        elif self.method == 'pcgrad':
            optimizer = PCGrad(self.optimizers())
            loss_0, loss_1, log_dict = self.compute_loss_pc(
                grasp_data.rotation,
                grasp_data.translation,
                grasp_data.sdf,
                grasp_data.mesh_path,
                grasp_data.normalization_scale,
                "train",
            )
            self.log_dict(
                log_dict,
                prog_bar=True,
                batch_size=self.config.data.batch_size,
            )
            losses = [loss_0, loss_1]
            optimizer.pc_backward(losses)
            optimizer.step()
        elif self.method == 'lsd':
            loss, log_dict = self.lsd_loss(
                grasp_data.rotation,
                grasp_data.translation,
                grasp_data.sdf,
                grasp_data.mesh_path,
                grasp_data.normalization_scale,
                "train",
            )
            self.log_dict(
                log_dict,
                prog_bar=True,
                batch_size=self.config.data.batch_size,
            )
            return loss
        else:
            print('method error')

    def validation_step(self, batch, batch_idx: int):
        # grasp_data = batch
        grasp_data = get_grasp_from_batch(batch)

        sdf_input = rearrange(grasp_data.sdf, "... -> 1 1 ...")

        so3_output, r3_output = sample(
            self.model,
            sdf_input,
            grasp_data.translation.device,
            torch.tensor(grasp_data.normalization_scale),
            self.config.training.num_samples_to_log,
            self.steps,
            sdf_path=grasp_data.mesh_path,
        )

        has_collision, scene, _, is_graspable = self.compute_grasp_scene(grasp_data, (r3_output, so3_output))
        self.all_collision.append(has_collision)
        self.all_graspable.append(is_graspable)
        # Log validation metrics

    def on_validation_epoch_end(self):
        print('val-start')
        success_all = []
        success_poss = 0
        for idx in range(len(self.all_collision)):
            success_20 = 0
            for i in range(len(self.all_collision[idx])):
                if not self.all_collision[idx][i]:
                    if self.all_graspable[idx][i]:
                        success_20 += 1
            if success_20 > 0:
                success_poss += 1
            success_all.append(success_20)

        # print('steps', self.steps)
        succ = 0
        if len(self.all_graspable) == 0:
            succ = 0
        else:
            succ = success_poss / len(self.all_graspable)
        print('steps', self.steps)
        print('---- 总体成功率 ----:', succ, ' len:', len(self.all_graspable))
        self.all_success.append(succ)
        self.all_collision = []
        self.all_graspable = []

        loss_dict = {
            f"val/so3_loss": succ,
            f"val/r3_loss": succ,
            f"val/loss": succ,
        }

        self.log_dict(
            loss_dict,
            prog_bar=True,
            batch_size=self.config.data.batch_size,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=tuple(self.config.training.adamw_betas),
            eps=self.config.training.epsilon,
            weight_decay=self.config.training.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        # Single linear scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.training.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.training.warmup_ratio,
            anneal_strategy="linear",
            div_factor=3.0,  # initial_lr = max_lr/div_factor
            final_div_factor=float("inf"),  # final_lr = initial_lr/final_div_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": self.config.training.checkpoint_metric,
            },
        }

    def on_validation_start(self) -> None:
        train_dataset = self.trainer.train_dataloader.dataset
        # Get base datasets (handle Subset case)
        train_base = (
            train_dataset.dataset
            if isinstance(train_dataset, torch.utils.data.Subset)
            else train_dataset
        )
        self.translation_norm_params = train_base.norm_params

    def on_train_start(self) -> None:
        """Setup logging of initial grasp scenes on training start."""
        print('----- ', self.method, ' -----')
        train_dataset = self.trainer.train_dataloader.dataset
        val_dataset = self.trainer.val_dataloaders.dataset

        # Get base datasets (handle Subset case)
        train_base = (
            train_dataset.dataset
            if isinstance(train_dataset, torch.utils.data.Subset)
            else train_dataset
        )
        val_base = (
            val_dataset.dataset
            if isinstance(val_dataset, torch.utils.data.Subset)
            else val_dataset
        )

        self.translation_norm_params = train_base.norm_params
        # print(self.translation_norm_params)

        # First get selected_indices from the base dataset if they exist
        base_selected = (
            train_base.selected_indices
            if hasattr(train_base, "selected_indices")
            else None
        )

        # Then get the actual split indices from Subset
        # if isinstance(train_dataset, torch.utils.data.Subset):
        #     train_indices = set(
        #         train_dataset.indices
        #     )  # These are indices into the base dataset
        #     val_indices = set(val_dataset.indices)
        # 
        #     # If base dataset had selected_indices, we need to map through them
        #     if base_selected is not None:
        #         train_indices = set(base_selected[i] for i in train_indices)
        #         val_indices = set(base_selected[i] for i in val_indices)
        # else:
        #     # If not a subset, use selected_indices directly if they exist
        #     train_indices = (
        #         set(train_base.selected_indices)
        #         if hasattr(train_base, "selected_indices")
        #         else None
        #     )
        #     val_indices = (
        #         set(val_base.selected_indices)
        #         if hasattr(val_base, "selected_indices")
        #         else None
        #     )
        # 
        # if train_indices is not None and val_indices is not None:
        #     if train_indices & val_indices:
        #         print(
        #             "Warning: Overlapping indices found between training and validation sets."
        #         )

        for prefix, dataset in [
            ("train", self.trainer.train_dataloader.dataset),
            ("val", self.trainer.val_dataloaders.dataset),
        ]:
            grasp_data = dataset[0]
            _, scene, _, _ = self.compute_grasp_scene(grasp_data)

            gripper_transform = torch.eye(4)
            gripper_transform[:3, :3] = grasp_data.rotation[:3, :3]
            gripper_transform[:3, 3] = denormalize_translation(
                grasp_data.translation, self.translation_norm_params
            ).squeeze()

            gripper_transform = wandb.Table(
                data=gripper_transform.cpu().numpy().tolist(),
                columns=["rot1", "rot2", "rot3", "tr"],
            )

            self.logger.experiment.log(
                {
                    f"{prefix}/original_grasp": scene_to_wandb_3d(scene),
                }
            )

    def on_train_end(self) -> None:
        print('success rate list:', self.all_success)
        print('----- ', self.method, ' -----')

    def duplicate_to_batch_size(self, input: Tensor, batch_size: int):
        current_size = input.size(0)
        if current_size >= batch_size:
            return input

        num_copies = batch_size // current_size
        remainder = batch_size % current_size

        duplicated = input.repeat(num_copies, *(1 for _ in range(len(input.shape) - 1)))
        if remainder > 0:
            duplicated = torch.cat([duplicated, input[:remainder]], dim=0)

        return duplicated

    def compute_grasp_scene(
            self,
            grasp_data: GraspData,
            r3_so3_inputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Get normalized translation and rotation from inputs or grasp_data
        normalized_translation, rotation = (
            r3_so3_inputs
            if r3_so3_inputs is not None
            else (grasp_data.translation, grasp_data.rotation)
        )

        # Denormalize and adjust translation with centroid
        denormalized_translation = denormalize_translation(
            normalized_translation, self.translation_norm_params
        )
        final_translation = denormalized_translation + torch.tensor(
            grasp_data.centroid, device=denormalized_translation.device
        )
        has_collision, scene, min_distance, is_graspable = check_collision(
            rotation,
            final_translation,
            grasp_data.mesh_path,
            grasp_data.dataset_mesh_scale,
        )
        # print('has_collision: ', has_collision)
        # print('is_graspable: ', is_graspable)
        return has_collision, scene, min_distance, is_graspable

    def test_step(self, batch, batch_idx: int):
        # 20个样本一个一个输入

        grasp_data = get_grasp_from_batch(batch)
        # grasp_data = batch

        sdf_input = rearrange(grasp_data.sdf, "... -> 1 1 ...")
        temp_collision = []
        temp_graspable = []
        for i in self.step_collec:
            so3_output, r3_output = sample(
                self.model,
                sdf_input,
                grasp_data.translation.device,
                torch.tensor(grasp_data.normalization_scale),
                self.config.training.num_samples_to_log,
                i,
                sdf_path=grasp_data.mesh_path,
            )
            # print("grasp_data", grasp_data)
            # print("r3_output", r3_output)
            # print("so3_output", so3_output)

            has_collision, scene, _, is_graspable = self.compute_grasp_scene(grasp_data, (r3_output, so3_output))
            temp_collision.append(has_collision)
            temp_graspable.append(is_graspable)
            self.logger.experiment.log(
                {
                    f"test/generated_grasp": scene_to_wandb_3d(scene),
                }
            )
        self.all_collision.append(temp_collision)
        self.all_graspable.append(temp_graspable)
        # all_collision = [[ []  ]     ]
        #                   batch_size个样本，每个样本里有7个len=20的数组，steps=1~7,每步采样20次

    def on_test_start(self):
        test_dataset = self.trainer.test_dataloaders.dataset

        # Get base datasets (handle Subset case)
        base = (
            test_dataset.dataset
            if isinstance(test_dataset, torch.utils.data.Subset)
            else test_dataset
        )
        self.translation_norm_params = base.norm_params

    def on_test_epoch_end(self) -> None:
        print('test-epoch-end')

    def on_test_end(self) -> None:

        for j in range(len(self.step_collec)):
            success_all = []
            success_poss = 0
            for idx in range(len(self.all_collision)):
                success_20 = 0
                for i in range(len(self.all_collision[idx][j])):
                    if not self.all_collision[idx][j][i]:
                        if self.all_graspable[idx][j][i]:
                            success_20 += 1
                if success_20 > 0:
                    success_poss += 1
                success_all.append(success_20)

            success = success_poss / len(self.all_graspable)
            print('method:',self.method ,', steps', self.step_collec[j], ', 总体成功率:', success, ', success_20: ',np.mean(success_all), '±', np.std(success_all))
            # print('---- 总体成功率 ----:', success_poss / len(self.all_graspable))
            # print('---- success_20 ----:', np.mean(success_all), '±', np.std(success_all))


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 'batch' should contain all grasps from one SDF, thanks to SingleSDFSampler.
        # Extract the actual data fields from your collated batch:
        print("batch_idx", batch_idx)
        grasp_data = get_grasp_from_batch(batch)

        # Here, `grasp_data.rotation` is all real rotations for that SDF,
        # `grasp_data.translation` is all real translations, etc.
        real_rotations = grasp_data.rotation
        print(real_rotations.shape)
        real_translations = grasp_data.translation
        real_sdf = grasp_data.sdf
        sdf_path = grasp_data.mesh_path  # Usually all are the same in one batch
        sdf_input = rearrange(real_sdf, "... -> 1 1 ...")
        print(sdf_input.shape, "sdf_input_size")
        # Generate synthetic grasps
        # (Example: sample 2000 predictions)
        so3_samples, r3_samples = sample(
            self.model,
            sdf_input,  # shape [1, ...]
            device=real_rotations.device,
            normalization_scale=torch.tensor(grasp_data.normalization_scale),
            num_samples=2,
            sdf_path=sdf_path,
        )

        # Compare distributions:
        from src.models.wasserstein import wasserstein_distance

        wdist_so3 = wasserstein_distance(so3_samples, real_rotations, space="so3")
        wdist_r3 = wasserstein_distance(r3_samples, real_translations, space="r3")

        self.log_dict(
            {
                "wdist_so3": wdist_so3,
                "wdist_r3": wdist_r3,
                "sdf_path": sdf_path,
            }
        )

        return {
            "sdf_path": sdf_path,
            "wdist_so3": wdist_so3,
            "wdist_r3": wdist_r3,
        }
