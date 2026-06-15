import os
os.environ['DISPLAY'] = ':11.0'
import torch
from src.data.util import denormalize_translation
from src.models.util import get_grasp_from_batch

if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import ExperimentConfig
    from src.core.visualize import check_collision
    from src.data.dataset import DataLoader, GraspDataset
    from src.data.util import GraspData

    config = ExperimentConfig.default_mlp()
    # /home/zhongzichen/code/RFM-Grasp-main/logs/checkpoints/run_20260312_205059/used_norm_params.pkl
    config.data.translation_norm_param_path = "logs/checkpoints/run_20260312_205059/used_norm_params.pkl"
    # config.data.translation_norm_param_path = "grasp_results/3837a34e62f54e5189f8950d7fb48ee2_scale_0.081.pkl"
    config.data.sample_limit = 100

    test = GraspDataset(
        data_root=config.data.data_path,
        config=config,
        grasp_files=config.data.files,
        num_samples=config.data.sample_limit,
        split="test",
    )

    batch_size = 1
    dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    # grasp_data: GraspData = test[0]
    batch = next(iter(dataloader))
    grasp_data = get_grasp_from_batch(batch)

    # Denormalize and adjust translation with centroid
    denormalized_translation = denormalize_translation(
        batch.translation, test.norm_params
    )
    final_translation = denormalized_translation + torch.tensor(
        grasp_data.centroid, device=denormalized_translation.device
    )

    # move = torch.tensor([0.011, 0, 0.001])  # barely colliding
    move = torch.tensor([-0.039, 0, 0])  # too far
    # move = torch.tensor([-0.025, 0, 0])  # graspable but almost too far
    # move = torch.tensor([0.011, 0, 0])  # very close
    # move = torch.tensor([0.0114, 0, 0])  # barely colliding
    final_translation = final_translation + move

    has_collision, scene, min_distance, is_graspable = check_collision(
        batch.rotation,
        final_translation,
        grasp_data.mesh_path,
        grasp_data.dataset_mesh_scale,
    )

    scene.show()

    # scene.export("logs/output.glb")
