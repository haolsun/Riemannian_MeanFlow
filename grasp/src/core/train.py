import atexit
from dataclasses import asdict

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from src.core.config import ExperimentConfig
from src.data.dataset import DataModule


def cleanup_wandb():
    """Ensure wandb run is properly closed."""
    try:
        wandb.finish()
    except:
        pass


def train(
    model: pl.LightningModule,
    config: ExperimentConfig,
):
    """Train the autoencoder model with improved logging and visualization."""
    # Register cleanup function
    atexit.register(cleanup_wandb)

    try:
        # Initialize WandB logger with modified settings
        wandb_logger = WandbLogger(
            project=config.training.project_name,
            name=config.training.run_name,
            save_dir=config.training.save_dir,
            settings=wandb.Settings(start_method="thread"),
        )

        # Log hyperparameters and config file
        wandb_logger.log_hyperparams(asdict(config))

        # Setup callbacks
        callbacks = []

        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.training.checkpoint_dir + "/" + config.training.run_name,
            filename=config.training.checkpoint_name,
            monitor=config.training.checkpoint_metric,
            mode=config.training.checkpoint_mode,
            save_last=config.training.save_last,
            save_top_k=config.training.keep_top_k_checkpoints,
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=config.training.checkpoint_metric,
            min_delta=config.training.early_stop_min_delta,
            patience=config.training.early_stop_patience,
            verbose=True,
            mode=config.training.checkpoint_mode,
            check_finite=True,  # Stop if loss becomes NaN or inf
        )
        callbacks.append(early_stopping)

        # Initialize trainer
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=config.training.max_epochs,
            accelerator="auto",
            #accelerator="cpu",
            devices="auto",
            precision=config.training.precision,
            gradient_clip_val=config.training.gradient_clip_val,
            accumulate_grad_batches=config.training.batch_accumulation,
            val_check_interval=config.training.validation_interval,
            check_val_every_n_epoch=config.training.val_every_n_epoch,
            log_every_n_steps=1,
        )

        # Add this right before trainer.fit()
        wandb.require("service")

        # Initialize data handler
        data_handler = DataModule(config)

        # Train model
        trainer.fit(model=model, datamodule=data_handler)
        # test model
        trainer.test(model=model, datamodule=data_handler)

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise

    finally:
        # Ensure wandb is properly closed
        cleanup_wandb()
        print("\nWandB run closed. Exiting...")
