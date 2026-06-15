if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models.lightning  import Lightning
    from src.core.train import train

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data = DataConfig.random_h5()
    config.data.split_ratio = 0.9995
    config.data.num_workers = 3
    print(config.data.files)
    # ---------------------

    # Sanity with 1
    # config.data.sample_limit = 1
    # config.data.batch_size = 1

    # Check if it can learn multiple samples
    # Starting with 4
    # config.data.sample_limit = 5
    # config.data.batch_size = 4

    # Now with 16
    # config.data.sample_limit = 18
    # config.data.batch_size = 16

    # config.training.r3_loss_weight = 5.0  # previously 3.0
    # config.training.so3_loss_weight = 1.0

    # Now with 128
    # config.data.sample_limit = 142  # 128 * 10/9 (batch size * splitratio)
    # config.data.batch_size = 128

    # ---------------------

    # Full one file
    config.data.sample_limit = None
    config.data.batch_size = 128
    config.data.dataset_workers = 12
    # NOTE: i set target_batch_size to 128
    # Which means training data is not duplicated anymore

    config.training.r3_loss_weight = 7.0  # previously 3.0
    config.training.so3_loss_weight = 1.0

    # ---------------------

    # Model
    config.model.hidden_dim = 1024
    config.training.early_stop_patience = 100
    config.training.max_epochs = 20000

    config.training.sample_interval = 200
    config.training.validation_interval = 0.025
    config.training.duplicate_ratio = 8
    # Initialize model
    config.data.files = 8000
    # model = Lightning.load_from_checkpoint(
    # checkpoint_path="logs/checkpoints/run_20250126_214412/model-epoch=66-val_loss=0.00.ckpt",
    # config=config,  
    # )

    model = Lightning.load_from_checkpoint(
    checkpoint_path="logs/checkpoints/run_20250201_224321/model-epoch=05-val_loss=0.00.ckpt",
    config=config,  
    )
    #model = Lightning(config)
    train(model, config)
