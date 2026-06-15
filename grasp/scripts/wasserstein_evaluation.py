if __name__ == "__main__":
    from scripts import initialize
    initialize()

    import json
    import os

    # Read the list of used files from JSON
    with open("logs/checkpoints/run_20250202_233846/used_grasp_files.json", "r") as f:
        used_files = json.load(f)

    # Get all files in the grasps directory
    all_files = os.listdir("data/grasps")

    # Find files that are not in the used_files list
    unused_files = [file for file in all_files if file not in used_files]

    print("Unused files:", len(unused_files))
    print("Used files:", len(used_files))
    print("All files:", len(all_files))

    from src.models.lightning import Lightning

    # Load the checkpoint and set the model to eval mode.
    model = Lightning.load_from_checkpoint(
        "logs/checkpoints/run_20250202_233846/last.ckpt"
    )
    model.eval()  # Note: Trainer.test/predict will set the model to eval mode anyway

    import pytorch_lightning as pl
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    from src.core.config import ExperimentConfig
    from src.data.dataset import DataModule

    # Create and adjust your experiment configuration
    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data.sample_limit = None
    config.data.files = unused_files
    config.data.translation_norm_param_path = "logs/checkpoints/run_20250204_204207/used_norm_params.pkl"
    config.data.dataset_workers = 8
    config.data.data_path = "data"

    # Finish any previous WandB run
    wandb.finish()

    # Initialize WandB logger with modified settings
    wandb_logger = WandbLogger(
        project=config.training.project_name,
        name=config.training.run_name,
        save_dir=config.training.save_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Setup callbacks if needed
    callbacks = []

    # Initialize Trainer (adjust accelerator/devices as needed)
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="cpu",
        devices="auto",
        log_every_n_steps=1,
    )

    # Initialize your DataModule (which should define predict_dataloader())
    data = DataModule(config)

    # Run prediction; note that we pass the DataModule so that its
    # predict_dataloader() method is called automatically.
    trainer.predict(model, datamodule=data)
