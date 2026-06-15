import os
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

translation_norm_params = 0



if __name__ == "__main__":


    from scripts import initialize

    initialize()

    from src.core.config import ExperimentConfig
    from src.data.dataset import GraspDataset, MeshBatchSampler
    from src.models.flow import sample
    from src.models.lightning import Lightning
    from src.models.util import get_grasp_from_batch
    from typing import Dict, Optional, Tuple


    def compute_grasp_scene(
            grasp_data: GraspDataset,
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

    # Get all files in the grasps directory
    all_files = os.listdir("data/grasp")
    # /home/zhongzichen/code/RFM-Grasp-main/logs/checkpoints/run_20260315_192912/model-epoch=189-val_loss=0.00.ckpt
    model = Lightning.load_from_checkpoint(
        "/home/zhongzichen/code/RFM-Grasp-main/logs/checkpoints/run_20260424_111222/model-epoch=04-val_loss=0.00.ckpt"
    )

    model.eval()

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data.sample_limit = None
    config.data.files = all_files[:2000]

    # config.data.files = [
    #     "Pizza_caca4c8d409cddc66b04c0f74e5b376e_0.0065985560890656995.h5",
    # ]

    config.data.translation_norm_param_path = "/home/zhongzichen/code/RFM-Grasp-main/logs/checkpoints/run_20260424_111222/used_norm_params.pkl"

    # config.data.translation_norm_param_path = (
    #     "logs/checkpoints/run_20250204_204207/used_norm_params.pkl"
    # )

    config.data.dataset_workers = 8
    config.data.data_path = "data/"

    # print('------- -------- --------', '\n', '------- executed --------', '\n','------- -------- --------')
    data = GraspDataset(
        data_root=config.data.data_path,
        grasp_files=config.data.files,
        config=config,
        num_samples=2000,
        device=model.device,
    )

    # print('------- -------- --------', '\n', '------- executed --------', '\n','------- -------- --------')
    # print('model device:', model.device)
    dl = DataLoader(
        dataset=data,
        batch_sampler=MeshBatchSampler(data),
        shuffle=False,
        persistent_workers=True,
        num_workers=1,
        generator=torch.Generator(device=model.device),
        batch_size=1,
    )

    # Create output directory
    output_dir = Path("grasp_results")
    output_dir.mkdir(exist_ok=True)

    duplicate_list = []
    all_collision = []
    all_graspable = []

    train_dataset = dl.dataset
    # Get base datasets (handle Subset case)
    train_base = (
        train_dataset.dataset
        if isinstance(train_dataset, torch.utils.data.Subset)
        else train_dataset
    )
    translation_norm_params = train_base.norm_params

    for batch in tqdm(dl, desc="Processing batches"):
        grasp_data = get_grasp_from_batch(batch)

        # Create a unique identifier using mesh path and normalization scale
        unique_id = (grasp_data.mesh_path, grasp_data.normalization_scale)

        # Skip if this combination already exists
        if unique_id in duplicate_list:
            # print(f"Skipping duplicate: {unique_id}")
            continue

        # Add to duplicate list
        duplicate_list.append(unique_id)

        sdf_input = rearrange(grasp_data.sdf, "... -> 1 1 ...")

        print("Sampling")

        so3_output, r3_output = sample(
            model.model,
            sdf_input,
            grasp_data.translation.device,
            torch.tensor(grasp_data.normalization_scale),
            num_samples=1024,
            sdf_path=grasp_data.mesh_path,
        )

        has_collision, scene, _, is_graspable = compute_grasp_scene(grasp_data, (r3_output, so3_output))
        all_collision.append(has_collision)
        all_graspable.append(is_graspable)

    success_all = []
    success_poss = 0
    for idx in range(len(all_collision)):
        success_20 = 0
        for i in range(len(all_collision[idx])):
            if not all_collision[idx][i]:
                if all_graspable[idx][i]:
                    success_20 += 1
        if success_20 > 0:
            success_poss += 1
        success_all.append(success_20)

    succ = 0
    if len(all_graspable) == 0:
        succ = 0
    else:
        succ = success_poss / len(all_graspable)
    print('---- 总体成功率 ----:', succ, ' len:', len(all_graspable))

