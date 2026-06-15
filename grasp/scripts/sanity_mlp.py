if __name__ == "__main__":
    from scripts import initialize

    initialize()

    from src.core.config import DataConfig, ExperimentConfig
    from src.core.train import train
    from src.models import lightning

    config: ExperimentConfig = ExperimentConfig.default_mlp()
    config.data = DataConfig.sanity()
    config.data.split_ratio = 0.99  # No validation set
    config.data.sample_limit = 10  # 100  # overfitting
    config.data.batch_size = 256
    config.model.hidden_dim = 512

    config.training.early_stop_patience = 10
    config.training.max_epochs = 100

    config.training.sample_interval = 1000

    # Initialize model
    model = lightning.Lightning(config)

    train(model, config)
