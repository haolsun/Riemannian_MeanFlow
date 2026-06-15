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

    # Print the unused files
    # print("Files not used in training:")
    # for file in unused_files:
    #     print(file)

    print(len(unused_files))
    print(len(used_files))
    print(len(all_files))

    from src.models.lightning import Lightning

    model = Lightning.load_from_checkpoint(
        "logs/checkpoints/run_20250202_233846/last.ckpt"
    )

    model.eval()

    import pytorch_lightning as pl
    import wandb
    from pytorch_lightning.loggers import WandbLogger

    from src.core.config import ExperimentConfig
    from src.data.dataset import DataModule

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data.sample_limit = None
    config.data.files = unused_files
    config.data.translation_norm_param_path = "logs/checkpoints/used_norm_params.pkl"
    #     config.data.translation_norm_param_path = (
    #     "logs/checkpoints/run_20250204_204207/used_norm_params.pkl"
    # )

    # config.data.files = [
    #     # "Pizza_caca4c8d409cddc66b04c0f74e5b376e_0.0065985560890656995.h5",
    #     # "Bottle_3108a736282eec1bc58e834f0b160845_0.014738534305634038.h5",
    #     # "Table_f81fd6b4376092d8738e43095496b061_0.006924702384677666.h5",
    #     # "Chair_6d6e634ff34bd350c511e6b9b3b344f3_0.0006261704047318024.h5",
    #     # "DiningTable_88e73431030e8494cc0436ebbd73343e_0.001291601827631892.h5",
    #     # "TvStand_3eefee315ac3db5dcc719373d4fe991c_0.001315898064683637.h5",
    #     # "FloorLamp_6ed99b140108856ed6f64cc59c2eb3d7_0.0032887769359107007.h5",
    #     # "2Shelves_df03b94777b1f0f9e4e3c2d62691be9_0.0016080441311659521.h5",
    # ]

    config.data.dataset_workers = 8
    config.data.data_path = "data"

    wandb.finish()
    # Initialize WandB logger with modified settings
    wandb_logger = WandbLogger(
        project=config.training.project_name,
        name=config.training.run_name,
        save_dir=config.training.save_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Setup callbacks
    callbacks = []

    # Initialize trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="cpu",
        devices="auto",
        log_every_n_steps=1,
    )

    data = DataModule(config)

    trainer.test(model, datamodule=data)
