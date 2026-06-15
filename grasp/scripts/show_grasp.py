if __name__ == "__main__":
    from scripts import initialize

    initialize()

    import torch

    from src.core.config import DataConfig
    from src.core.visualize import check_collision
    from src.data.dataset import GraspDataset

    print(f"CUDA is available: {torch.cuda.is_available()}")

    # If CUDA is available, you can also print additional information
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")

    exit()

    config = DataConfig.sanity()

    test = GraspDataset(
        data_root=config.data_path,
        grasp_files=config.files,
        num_samples=config.sample_limit,
        split="test",
    )

    (
        so3_input,
        r3_input,
        norm_params,
        _,  # sdf_input
        mesh_path,
        dataset_mesh_scale,
        normalization_scale,
    ) = test[0]

    so3_input = torch.tensor(
        [
            [-0.2121, 0.3016, 0.9295],
            [-0.1925, 0.9196, -0.3423],
            [
                -0.9581,
                -0.2516,
                -0.137,
            ],
        ]
    )

    r3_input = torch.tensor([1.602, 0.5431, 0.3002])

    print("SO3 Input:", so3_input)
    print("R3 Input:", r3_input)
    print("Mesh Path:", mesh_path)
    print("Dataset Mesh Scale:", dataset_mesh_scale)

    has_collision, scene, min_distance, is_graspable = check_collision(
        so3_input,
        r3_input,
        mesh_path,
        dataset_mesh_scale,
    )

    # Print collision status
    print(f"Collision: {has_collision}")
    print(f"Minimum Distance: {min_distance}")

    # Show the scene
    # scene.show()

    scene.export("logs/output.glb")

    # img = scene_to_wandb_image(scene)
    # img.image.show()
