if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models import lightning
    from src.core.train import train
    import pytorch_lightning as pl

    pl.seed_everything(43) # 40 41 42 43 44

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    # config.data = DataConfig.small_one_file()
    config.data = DataConfig.random_h5_1000()
    # print(config.data)
    config.training.early_stop_min_delta = 1e-5
    config.training.early_stop_patience = 200
    config.training.max_epochs = 10
    config.training.steps = 1
    config.training.r3_loss_weight = 7.0  # previously 3.0
    config.training.so3_loss_weight = 1.0
    config.data.num_workers = 3
    config.training.method = 'mf'

    config.model.hidden_dim = 1024
    # config.model.num_hidden_layers = 10
    config.training.early_stop_patience = 100

    config.training.sample_interval = 10000
    config.training.validation_interval = 0.999999
    config.training.duplicate_ratio = 8

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
