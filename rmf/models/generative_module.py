"""
General generative module base classes.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Callable

from torchmetrics import MeanMetric
from wandb import Image

from lightning import LightningModule
import torch
from torch import Tensor, nn
from torchvision.utils import make_grid
from torchdiffeq import odeint

import geopandas
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point

from gfm.manifold.manifold import Manifold, ManifoldType, manifold_from_name
from gfm.data.components.util import cartesian_from_latlon, lonlat_from_cartesian
from gfm.models.pcgrad import PCGrad
from scipy.interpolate import make_interp_spline
from gfm.manifold import SO3
from scipy.spatial.transform import Rotation
# plt.rcParams['text.usetex'] = True

def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = torch.func.jacrev(u)

    def ret(x):
        return torch.trace(J(x).squeeze())

    return ret


def output_and_div(vecfield, x, v=None, div_mode="exact"):
    if div_mode == "exact":
        dx = vecfield(x)
        div = torch.vmap(div_fn(vecfield))(x)
    else:
        dx, vjpfunc = torch.func.vjp(vecfield, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


class _ManifoldProjected(nn.Module):
    """
    A wrapper for a neural network that projects its output to the tangent space of a manifold.

    :param net: the neural network to wrap.
    :param manifold: the manifold to project to.
    :param metric_normalize: whether to apply metric normalization to the vector field.
    """

    def __init__(self, net: nn.Module, manifold: Manifold, metric_normalize: bool):
        super().__init__()
        self.net = net
        self.manifold = manifold
        self.metric_normalize = metric_normalize

    def forward(self, x: Tensor, r: Tensor, t: Tensor) -> Tensor:
        # a bit ugly, but useful for SO3
        if x.ndim == 3 and self.manifold.get_type() == ManifoldType.SO3:
            out = self.net(x.reshape(-1, 9), r, t)
        else:
            out = self.net(x, r, t)

        if self.manifold.get_type() == ManifoldType.SO3:
            out = self.manifold.project_tangent(x.reshape(-1, 3, 3), out.reshape(-1, 3, 3))
        else:
            out = self.manifold.project_tangent(x, out)
        if self.metric_normalize:
            out = self.manifold.metric_normalize(x, out)
        return out.reshape_as(x)


class GenerativeModule(LightningModule, ABC):
    def __init__(
        self,
        net: nn.Module,
        in_shape: tuple[int, ...],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        compile: bool,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        div_mode: str = "exact",
        epoch = 0,
        my_method='alpha-p',
        val_step=8,
        so3_plot= False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        torch.set_float32_matmul_precision("high")
        self.save_hyperparameters(logger=True, ignore=["teacher"])
        self.net = net
        self._val_nll_metric = MeanMetric() if self.val_nll else None
        self._test_nll_metric = MeanMetric() if self.val_nll else None

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_number = 0
        self.val_nll_old = 0
        self.val_loss_old = 0
        self.epoch = epoch
        self.my_method = my_method
        self.val_step = val_step
        self.mmd_list = []
        self.negative = 0
        self.positive = 0
        self.grad_cos = []
        self.mean_list = []
        self.so3_plot = so3_plot

        # print('-----------------',epoch,'----------------')

    def get_instantaneous_vf(self, t: Tensor) -> Callable[[Tensor], Tensor]:
        """
        Returns a callable that computes the instantaneous vector field at time `t`.

        :param t: time tensor of shape (B,).
        :return: the instantaneous vector field at time `t`.
        """
        return lambda x: self.net(x, t, t)

    def get_stage(self) -> str:
        """
        Returns whether the trainer is in train/val/test.
        """
        if self.trainer.state.stage == "train":
            return "train"
        elif self.trainer.state.stage == "validate":
            return "val"
        elif self.trainer.state.stage in ["test", "predict"]:
            return "test"
        return "misc"

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the method.
        """

    @abstractmethod
    def get_loss(self, x: Tensor | list[Tensor] | dict[str, Tensor]) -> Tensor | None:
        """
        Computes the loss.
        """

    @abstractmethod
    def get_loss_imf(self, x: Tensor | list[Tensor] | dict[str, Tensor]) -> Tensor | None:
        """
        Computes the loss.
        """

    @abstractmethod
    def get_loss_alpha_p(self, x: Tensor | list[Tensor] | dict[str, Tensor]) -> Tensor | None:
        """
        Computes the loss.
        """

    @abstractmethod
    def get_loss_pc_grad(self, x: Tensor | list[Tensor] | dict[str, Tensor]):
        """
        Computes the loss.
        """

    @abstractmethod
    def get_loss_alpha_flow(self, x: Tensor | list[Tensor] | dict[str, Tensor]) -> Tensor | None:
        """
        Computes the loss.
        """

    @property
    def is_sampling_time_zero_to_one(self) -> bool:
        """
        Returns `True` iff the sampling process starts at time zero, and ends at time one.
        """
        return True

    @property
    def val_nll(self) -> bool:
        """
        True if negative log likelihood should be evaluated at the end of each epoch
        on each validation batch, and on each test batch.
        """
        return False

    @property
    def val_images(self) -> bool:
        """
        True if images should be sampled at the end of each epoch.
        """
        return False

    @abstractmethod
    def sample_batch(self, steps: int, sz: int) -> Tensor:
        """
        Samples a batch of size `sz` in `steps` steps.

        :param steps: number of steps to sample with.
        :param sz: number of samples to generate.
        """

    @abstractmethod
    def log_prob_prior(self, x: Tensor) -> Tensor:
        """
        Computes the log probability of the prior distribution at the given points.
        """

    @torch.inference_mode(False)
    def sample(
        self, steps: int, sz: int, batch_size: int | None = None
    ) -> list[Tensor]:
        """
        Samples a batch of size `sz` in `steps` steps.

        :param steps: number of steps to sample with.
        :param sz: total number of samples to generate.
        :param batch_size: batch size for sampling. If None, uses the training batch size.
        :return: list of tensors of samples.
        """
        to_sample = sz
        batch_size = batch_size or self.trainer.datamodule.hparams.batch_size
        ret = []
        while to_sample > 0:
            samples = self.sample_batch(steps=steps, sz=min(batch_size, to_sample))
            to_sample -= samples.shape[0]
            ret += [samples]
        return ret

    @torch.inference_mode(False)  # to use torch.func
    def evaluate_nll(self, x0: Tensor) -> Tensor:
        """
        Evaluates the negative log likelihood of the model on the input data `x`.

        From: https://github.com/facebookresearch/riemannian-fm.
        """
        v = None
        if self.hparams.div_mode == "rademacher":
            v = torch.randint(low=0, high=2, size=x0.shape).to(x0) * 2 - 1

        def odefunc(t, tensor):
            t = t.to(tensor)
            x = tensor[..., : self.hparams.in_shape[-1]]
            vecfield = self.get_instantaneous_vf(t)
            dx, div = output_and_div(vecfield, x, v=v, div_mode=self.hparams.div_mode)

            """
            if hasattr(self.manifold, "logdetG"):

                def _jvp(x, v):
                    return torch.func.jvp(self.manifold.logdetG, (x,), (v,))[1]

                corr = torch.vmap(_jvp)(x, dx)
                div = div + 0.5 * corr.to(div)
            """

            div = div.reshape(-1, 1)
            del t, x
            return torch.cat([dx, div], dim=-1)

        # TODO: only for SO(3), might need fix
        if x0.ndim > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        state0 = torch.cat([x0, torch.zeros_like(x0[..., :1])], dim=-1)
        if self.is_sampling_time_zero_to_one:
            timesteps = torch.linspace(1.0, 0.0, 2)
        else:
            timesteps = torch.linspace(0.0, 1.0, 2)
        time_timp = timesteps.to(x0)
        try:
            state1 = odeint(
                odefunc,
                state0,
                t=timesteps.to(x0),
                atol=self.hparams.atol,
                rtol=self.hparams.rtol,
                method="dopri5",
                options={"min_step": 1e-5},
            )[-1]
            x1, logdetjac = state1[..., : self.hparams.in_shape[-1]], state1[..., -1]
            if hasattr(self, "manifold"):
                x1 = self.manifold.project(x1)
            logp1 = self.log_prob_prior(x1)
            ll = logp1 + logdetjac
            return -ll
        except Exception as e:
            print(f"Encountered error during NLL eval: {e}")
            return torch.full((x0.shape[0],), torch.inf).to(x0)


    @torch.inference_mode()
    def _compute_mmd(self, x: Tensor, y: Tensor, gamma: float = 1.0, batch_size: int | None = None) -> float:
        assert x.shape[0] == y.shape[0]

        def _kernel(a: Tensor, b: Tensor) -> Tensor:
            ds = self.manifold.distance(a, b).square_()
            ds = (-gamma * ds).exp()
            return ds

        # calculate inner x, inner y, cross x and y
        x0 = x.repeat((x.shape[0],) + (1,) * (x.ndim - 1))
        x1 = x.repeat_interleave(x.shape[0], dim=0)
        y0 = y.repeat((y.shape[0],) + (1,) * (y.ndim - 1))
        y1 = y.repeat_interleave(y.shape[0], dim=0)
        inner_x = _kernel(x0, x1)
        inner_y = _kernel(y0, y1)
        cross_xy = _kernel(x0, y1)
        mmd = (inner_x + inner_y - 2 * cross_xy).mean().sqrt().item()
        del inner_x, inner_y, cross_xy, x0, y0, x1
        self.mmd_list.append(mmd)
        return mmd

    @torch.inference_mode(False)
    def _evaluate_mmd(self, stage: str) -> None:
        assert stage in ["val", "test"], "can only be in val/train for MMD evaluation"
        if not hasattr(self.trainer.datamodule, f"get_{stage}_tensor"):
            print(
                f"No get_{stage}_tensor defined on datamodule, {type(self.trainer.datamodule).__name__}, skipping MMD evaluation"
            )
            return
        # access validation set
        try:
            val_set: Tensor = getattr(
                self.trainer.datamodule, f"get_{stage}_tensor"
            )().to(self.device)
            steps = [1, 2, 4]
            for step in steps:
                samples = self.sample(steps=step, sz=val_set.shape[0])
                samples = torch.cat(samples, dim=0)
                # compute MMD
                mmd = self._compute_mmd(samples, val_set)
                del samples
                self.log(
                    f"{stage}/mmd-{step:03d}",
                    mmd,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                )
        except Exception as e:
            print("Exception encountered during MMD evaluation:", e)

    def log_train(self, loss):
        self.train_loss(loss or torch.inf)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )

    def training_step(self, batch: Tensor | dict[str, Tensor], batch_idx: int):
        if self.my_method == 'alpha-p':
            loss = self.get_loss_alpha_p(batch)
            self.log_train(loss)
            return loss
        elif self.my_method == 'pcgrad':
            # self.optimizers().zero_grad()
            # loss, loss_2 = self.get_loss_pc_grad(batch)
            # # print('raw loss:',loss, flush=True)
            # self.backward(loss=loss, retain_graph=True)
            # grad_0 = []
            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         # print(name, param.grad.mean(), flush=True)
            #         grad_0.append(param.grad.mean().item())
            # # print('grad_0:', grad_0, flush=True)
            # grad_0 = torch.tensor(grad_0).to(self.device)
            # self.backward(loss=loss_2)
            # grad_1 = []
            # for name, param in self.named_parameters():
            #     if param.grad is not None:
            #         # print(name, param.grad.mean(), flush=True)
            #         grad_1.append(param.grad.mean())
            # grad_1 = torch.tensor(grad_1).to(self.device)
            # # print('grad_1:', grad_1, flush=True)
            # if grad_0.shape[0] > 0 and grad_1.shape[0] > 0:
            #     cos_sim = torch.nn.functional.cosine_similarity(grad_0, (grad_1 - grad_0), dim=0)
            #     self.grad_cos.append(cos_sim.cpu())
            #     self.mean_list.append(np.mean(self.grad_cos))
            #     if cos_sim > 0:
            #         self.positive += 1
            #     else:
            #         self.negative += 1
            #     # print('positive:', self.positive, ' | negative:', self.negative)
            # self.optimizers().step()
            optimizer = PCGrad(self.optimizers())
            loss_0, loss_1 = self.get_loss_pc_grad(batch)
            self.log_train(loss_0 + loss_1)
            losses = [loss_0, loss_1]
            optimizer.pc_backward(losses)
            optimizer.step()
        elif self.my_method == 'alpha-f':
            loss = self.get_loss_alpha_flow(batch)
            self.log_train(loss)
            return loss
        elif self.my_method == 'mf':
            loss = self.get_loss(batch)
            self.log_train(loss)
            return loss
        elif self.my_method == 'imf':
            loss = self.get_loss_imf(batch)
            self.log_train(loss)
            return loss
        else:
            print("---------- unknown method  ----------")


    def validation_step(self, batch: Tensor | dict[str, Tensor], batch_idx: int):
        # if (self.val_number + 1) % self.val_step != 0:
        #     loss = self.val_loss_old
        #     nll = self.val_nll_old
        # else:
        #     loss = self.get_loss(batch)
        #     self.val_loss_old = loss
        #     if self.val_nll:
        #         if isinstance(batch, dict):
        #             nll = self.evaluate_nll(batch["x1"])
        #         else:
        #             nll = self.evaluate_nll(batch)
        #         self.val_nll_old = nll
        # self.val_number += 1
        #
        # if loss is not None:
        #     self.val_loss(loss)
        # self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # if self.val_nll:
        #     self._val_nll_metric(nll)
        #     self.log(
        #         "val/nll",
        #         self._val_nll_metric,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #     )
        if self.hparams.evaluate_mmd and self.current_epoch % 2 == 0:
            self._evaluate_mmd("val")

        if self.current_epoch % 20 == 0:
            y = self.grad_cos
            x = [i for i in range(len(self.grad_cos))]
            plt.rcParams.update({'font.size': 18})
            plt.rcParams["font.family"] = "Times New Roman"
            plt.figure()
            plt.plot(x, y, label=r'cos(▽L_1, ▽L_2)', color='mistyrose')
            mean_y = self.mean_list
            plt.plot(x, mean_y, label=r'mean_cos(▽L_1, ▽L_2)', color='r')
            plt.legend(loc="upper right")
            plt.xlabel('Iteration', fontsize=18)
            plt.ylabel('Cosine Similarity', fontsize=18)
            plt.axhline(y=0, color='black',linestyle=(0, (5, 2)))
            plt.savefig('/home/zhongzichen/code/gfm-main/figs/cos_chart'+ str(len(y)) +'.png', bbox_inches='tight')
            # plt.show()


    def test_step(self, batch: Tensor | dict[str, Tensor], batch_idx: int):
        loss = self.get_loss(batch)
        self.test_loss(loss or torch.inf)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        # if False:
        #     nll = self.evaluate_nll(batch)
        #     self._test_nll_metric(nll)
        #     self.log(
        #         "test/nll",
        #         self._test_nll_metric,
        #         on_step=False,
        #         on_epoch=True,
        #         prog_bar=True,
        #         logger=True,
        #     )

    def on_validation_epoch_end(self) -> None:
        if self.val_images:
            batch = self.sample_batch(steps=1, sz=64)
            # make grid of samples torchvision
            grid = make_grid(
                batch,
                nrow=8,
                normalize=True,
                value_range=self.trainer.datamodule.value_range,
            )
            img = Image(grid)
            self.logger.experiment.log(
                {
                    "val/samples": img,
                    "global_step": self.current_epoch,
                }
            )

    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class RiemannianGenerativeModule(GenerativeModule):
    """
    Generative module for Riemannian geometry.

    :param evaluate_mmd (list[int] | None): None if the MMD should not be evaluated. If
        not none, then the MMD for the specified steps will be evaluated.
    :param tori_plot (bool): whether to plot torus plots at the end of testing.
    :param earth_plot (bool): whether to plot earth plots at the end of testing.
    :param skip_proj (bool): whether to skip projection to the manifold tangent space.
    :param metric_normalize (bool): whether to apply metric normalization to the vector field.
    """

    def __init__(
        self,
        manifold: str = "euclidean",
        evaluate_mmd: list[int] | None = [1, 2, 8, 16, 32, 64, 100],
        tori_plot: bool = False,
        earth_plot: bool = False,
        skip_proj: bool = False,
        metric_normalize: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.manifold = manifold_from_name(manifold)
        self.net = _ManifoldProjected(self.net, self.manifold, metric_normalize) if not skip_proj else self.net

    def log_prob_prior(self, x: Tensor) -> Tensor:
        return self.manifold.logp0(x)

    def on_test_epoch_end(self) -> None:
        if self.hparams.evaluate_mmd:
            self._evaluate_mmd("test")

        # batch = self.sample(steps=1, sz=20000, batch_size=20000)
        # batch = batch[0].view(-1, 9)
        # # self.plot_so3(batch_1[0], 'cone')
        # with open('/data1/zzc/code/gfm-main/gfm/models/sample_data.txt', 'w') as f:
        #     for row in batch:
        #         f.write(','.join(map(str, row.tolist())) + '\n')

        # rama plots
        if self.hparams.tori_plot:
            batch = self.trainer.datamodule.get_test_tensor()
            self.plot_torus2d(batch)

        if self.hparams.earth_plot:
            batch = self.trainer.datamodule.get_test_tensor()
            self.plot_earth2d(batch)
        if self.so3_plot:
            batch = self.sample(steps=1, sz=20000, batch_size=20000)
            self.plot_so3(batch[0])

    # def on_validation_epoch_end(self) -> None:
    #     if self.hparams.evaluate_mmd and self.current_epoch % 10 == 0:
    #         self._evaluate_mmd("val")

    # From Riemannian-FM, META, all rights reserved, etc.

    def on_train_start(self) -> None:
        if self.my_method == 'alpha-p':
            print("---------- alpha-parameter   ----------")
        elif self.my_method == 'pcgrad':
            print("---------- PCGrad  ----------")
        elif self.my_method == 'alpha-f':
            print("---------- alpha-flow  ----------")
        elif self.my_method == 'mf':
            print("---------- mean flow or flow map ----------")
        elif self.my_method == 'imf':
            print("---------- improved mean flow  ----------")
        else:
            print("---------- unknown  ----------")

    def on_train_end(self) -> None:
        if self.my_method == 'alpha-p':
            print("---------- alpha-parameter   ----------")
        elif self.my_method == 'pcgrad':
            print("---------- PCGrad  ----------")
        elif self.my_method == 'alpha-f':
            print("---------- alpha-flow  ----------")
        elif self.my_method == 'mf':
            print("---------- mean flow or flow map ----------")
        elif self.my_method == 'imf':
            print("---------- improved mean flow  ----------")
        else:
            print("---------- unknown  ----------")


    def apply_rotvec(self, x, rotvec):
        theta = torch.norm(rotvec, dim=-1, p=2, keepdim=True)
        sin, cos = torch.sin(theta), torch.cos(theta)
        axis = rotvec / theta.clamp(min=1e-4)
        ad = torch.cross(axis, x, dim=-1)
        ad2 = torch.cross(axis, ad, dim=-1)
        return x + sin * ad + (1 - cos) * ad2

    # come from Riemannian Consistency Model
    def plot_so3(self, batch) -> None:
        # 3 * 3 --> 1 * 3
        batch = Rotation.from_matrix(batch).as_rotvec()
        batch = torch.from_numpy(batch).float()

        x_axis, y_axis, z_axis = torch.eye(3, dtype=torch.float).unsqueeze(1)
        x_rotated = self.apply_rotvec(x_axis, batch).numpy()
        y_rotated = self.apply_rotvec(y_axis, batch).numpy()
        z_rotated = self.apply_rotvec(z_axis, batch).numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='w', alpha=0.3)

        ax.quiver(0, 0, 0, 1, 0, 0, color='black', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='black', arrow_length_ratio=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='black', arrow_length_ratio=0.1)
        ax.text(1.1, 0, 0, 'X', color='black')
        ax.text(0, 1.1, 0, 'Y', color='black')
        ax.text(0, 0, 1.1, 'Z', color='black')

        ax.scatter(x_rotated[:, 0], x_rotated[:, 1], x_rotated[:, 2], color='red', s=10, label='x-axis rotated',
                   alpha=0.1)
        ax.scatter(y_rotated[:, 0], y_rotated[:, 1], y_rotated[:, 2], color='green', s=10, label='y-axis rotated',
                   alpha=0.1)
        ax.scatter(z_rotated[:, 0], z_rotated[:, 1], z_rotated[:, 2], color='blue', s=10, label='z-axis rotated',
                   alpha=0.1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20., azim=50, roll=0)

        ax.legend()
        plt.savefig('so3.png')
        # plt.show()


    @torch.inference_mode(False)
    def plot_earth2d(self, batch):
        """
        Earth plots. From https://github.com/facebookresearch/riemannian-fm.
        """
        os.makedirs("figs", exist_ok=True)

        # Plot world map
        world = geopandas.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        ax = world.plot(figsize=(9, 4), antialiased=False, color="grey")

        # Plot model samples
        # samples = self.sample(batch.shape[0], batch.device)
        # samples = samples.cpu()
        # geometry = [Point(lonlat_from_cartesian(x) / np.pi * 180) for x in samples]
        # pts = geopandas.GeoDataFrame(geometry=geometry)
        # pts.plot(ax=ax, color="#1a9850", markersize=0.01, alpha=0.7)

        # Plot model likelihood
        N = 400
        x = np.linspace(-180.0, 180.0, N)  # longitude
        y = np.linspace(-90.0, 90.0, N)  # latitude
        X, Y = np.meshgrid(x, y)

        folder = f"figs/{self.name()}/"
        os.makedirs(folder, exist_ok=True)

        if os.path.exists(
            f"{folder}/{self.trainer.datamodule.hparams.dataset_file}-logps-{N}.npy"
        ):
            L = np.load(
                f"{folder}/{self.trainer.datamodule.hparams.dataset_file}-logps-{N}.npy"
            )
        else:
            lonlat = np.stack([Y.reshape(-1), X.reshape(-1)], axis=-1)
            xyz = cartesian_from_latlon(torch.tensor(lonlat) * np.pi / 180)
            logps = []
            for c in tqdm(torch.split(xyz, 8000)):
                c = c.to(batch).to(self.device)
                logps.append(-self.evaluate_nll(c).cpu().numpy())
            logps = np.concatenate(logps, axis=0)
            L = logps.reshape(N, N)
            np.save(
                f"{folder}/{self.trainer.datamodule.hparams.dataset_file}-logps-{N}.npy",
                L,
            )

        P = np.exp(L)
        cs = ax.contourf(
            X,
            Y,
            P,
            levels=np.linspace(0, 1, 11),
            alpha=0.7,
            extend="max",
            cmap="BuGn",
            antialiased=True,
        )

        # Plot data samples
        batch = batch.cpu()
        geometry = [Point(lonlat_from_cartesian(x) / np.pi * 180) for x in batch]
        pts = geopandas.GeoDataFrame(geometry=geometry)
        pts.plot(ax=ax, color="#d73027", markersize=0.01, alpha=0.7)

        cbar = plt.colorbar(cs, ax=ax, pad=0.01, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "$\geq$1"])
        cbar.ax.set_ylabel("likelihood", fontsize=18, rotation=270, labelpad=10)
        ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.set_alpha(0.7)

        # plt.axis("off")
        plt.xlim([-180, 180])
        plt.ylim([-90, 90])
        plt.xlabel("Longitude", fontsize=18)
        plt.ylabel("Latitude", fontsize=18)
        plt.tight_layout()
        plt.savefig(
            f"{folder}/{self.trainer.datamodule.hparams.dataset_file}-samples-{self.global_step:06d}.png",
            dpi=300,
        )
        plt.savefig(
            f"{folder}/{self.trainer.datamodule.hparams.dataset_file}-samples-{self.global_step:06d}.pdf"
        )
        plt.close()

    @torch.inference_mode(False)
    def plot_torus2d(self, batch):
        """
        Rama plots. From https://github.com/facebookresearch/riemannian-fm.
        """
        os.makedirs("figs", exist_ok=True)

        plt.rcParams["axes.autolimit_mode"] = "round_numbers"

        plt.figure(figsize=(6.1, 5))
        ax = plt.gca()

        # Plot model samples
        # samples = self.sample(batch.shape[0], batch.device)
        # samples = samples.cpu().numpy()
        # plt.scatter(samples[..., 0], samples[..., 1], marker=".", c="C0", s=1)

        # Plot density
        N = 400
        x = np.linspace(-np.pi, np.pi, N)  # longitude
        y = np.linspace(-np.pi, np.pi, N)  # latitude
        X, Y = np.meshgrid(x, y)

        folder = f"figs/{self.name()}/"

        if os.path.exists(
            f"{folder}{self.trainer.datamodule.hparams.data_type}-logps-{N}.npy"
        ):
            L = np.load(
                f"{folder}{self.trainer.datamodule.hparams.data_type}-logps-{N}.npy"
            )
        else:
            os.makedirs(folder, exist_ok=True)
            inputs = np.stack([X.flatten(), Y.flatten()], axis=-1)
            # wrap to [0, 2pi]
            inputs = torch.tensor(inputs + np.pi).to(self.device, dtype=torch.float32)
            logps = []
            for c in tqdm(torch.split(inputs, 8000)):
                logps.append(-self.evaluate_nll(c).cpu().numpy())
            logps = np.concatenate(logps, axis=0)
            L = logps.reshape(N, N)
            np.save(
                f"{folder}{self.trainer.datamodule.hparams.data_type}-logps-{N}.npy", L
            )

        X = X / np.pi * 180
        Y = Y / np.pi * 180
        cs = ax.contourf(X, Y, L, alpha=0.9, cmap="Blues", antialiased=True)

        # Plot data samples
        batch = batch.cpu().numpy()[:10000]
        # batch = (batch + np.pi) % (2 * np.pi) - np.pi
        # batch = batch / np.pi * 180
        batch = (batch - np.pi) / np.pi * 180

        plt.scatter(
            batch[..., 0], batch[..., 1], marker=".", c="#d73027", s=0.05, alpha=0.7
        )
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        ax.set_aspect("equal")
        plt.xlabel(r"$\phi$", fontsize=18)
        plt.ylabel(r"$\psi$", fontsize=18, rotation=0)

        plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.8, linewidth=0.5)
        plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.8, linewidth=0.5)

        cbar = plt.colorbar(cs, ax=ax, pad=0.01)
        cbar.ax.set_ylabel("log likelihood", fontsize=18, rotation=270, labelpad=10)
        ax.tick_params(axis="both", which="both", direction="in", length=3)
        cbar.ax.tick_params(axis="both", which="both", direction="in", length=3)

        plt.tight_layout()
        plt.savefig(f"{folder}{self.trainer.datamodule.hparams.data_type}.pdf")
        plt.close()
