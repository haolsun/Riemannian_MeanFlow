if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.models import lightning
    from src.core.train import train
    import pytorch_lightning as pl

    pl.seed_everything(41) # 40 41 42 43 44

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    # config.data = DataConfig.small_one_file()
    config.data = DataConfig.random_h5()
    # print(config.data)
    config.training.early_stop_min_delta = 1e-5
    config.training.early_stop_patience = 200
    config.training.max_epochs = 20
    config.training.steps = 1
    config.training.method = 'lsd'

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
