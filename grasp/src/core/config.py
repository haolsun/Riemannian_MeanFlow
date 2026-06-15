from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal
from pathlib import Path
import random



import torch


@dataclass
class MLPModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    sigma_min: float = 1e-4
    activation = torch.nn.ReLU
    num_hidden_layers: int = 4
    voxel_output_size: int = 256

    @classmethod
    def default(cls) -> "MLPModelConfig":
        return cls(
            input_dim=12,
            output_dim=12,
            hidden_dim=512,
        )


@dataclass
class DataConfig:
    data_path: str
    files: list[str]
    #sampler_opt: str
    batch_size: int = 32
    num_workers: int = 4
    sample_limit: Optional[int] = None
    split_ratio: float = 0.9  # Train-Val split ratio of 90-10%
    dataset_workers: int = 16
    translation_norm_param_path: Optional[str] = None
    

    @classmethod
    def sanity(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["AccentChair_3bc766749e4e4a7fe3f7a74e12a274ef_2.3008213565383318e-05.h5"],
            batch_size=8,
            sample_limit=1,
            #sampler_opt="repeat",
        )

    @classmethod
    def small_one_file(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_14e5dba73b283dc7fe0939859a0b15ea_0.0005312646125977.h5"],
            batch_size=8,
            sample_limit=10,
            split_ratio=0.8,
            #sampler_opt="repeat",
        )
        
    @classmethod
    def two_files(cls) -> "DataConfig":
        return cls(
            data_path="data/",
            files=["Xbox360_3837a34e62f54e5189f8950d7fb48ee2_0.00027076605672574777.h5",
                   "Xbox360_33c76a59e0f2a38cbf4723a8638af381_0.005158106214319449.h5"],
            batch_size=8,
            sample_limit=10,
            split_ratio=0.8,
            #sampler_opt="repeat",
        )
    

    @classmethod
    def random_h5(cls) -> "DataConfig":
        # random.seed(42)  # Fix the seed for reproducibility
        path = r'data/grasp'
        source_grasps = Path(path)
        all_h5 = [i.name for i in source_grasps.glob(r"*.h5")]
        random.shuffle(all_h5)
        selected = all_h5[:200]
        return cls(
            data_path="data/",
            files=selected,
            batch_size=128,
            split_ratio=0.9999,
            #sampler_opt="repeat",
        )

    @classmethod
    def random_h5_1000(cls) -> "DataConfig":
        # random.seed(42)  # Fix the seed for reproducibility
        path = r'data/grasp'
        source_grasps = Path(path)
        all_h5 = [i.name for i in source_grasps.glob(r"*.h5")]
        random.shuffle(all_h5)
        selected = all_h5[:2000]
        return cls(
            data_path="data/",
            files=selected,
            batch_size=256,
            split_ratio=0.999999,
            # sampler_opt="repeat",
        )

@dataclass
class TrainingConfig:
    """Consolidated training configuration"""

    # Training parameters
    steps: int = 1
    method = 'pcgrad'
    max_epochs: int = 100
    precision: Literal[16, 32, 64] = 32
    batch_accumulation: int = 1
    gradient_clip_val: float = 1.0
    r3_loss_weight: float = 3.0
    so3_loss_weight: float = 1.0
    duplicate_ratio: int = 1

    # Optimizer & Scheduler
    learning_rate: float = 1e-4
    weight_decay: float = 3e-9
    min_learning_rate: float = 1e-6
    scheduler_steps: int = 2000
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    warmup_ratio: float = 0.1

    # Validation and Logging
    validation_interval: int = 0.999999
    val_every_n_epoch: int = 1
    num_samples_to_log: int = 20
    sample_interval: int = 1000
    test_wasserstein: bool = False
    test_sample_success: bool = False

    # Checkpointing
    checkpoint_dir: str = "logs/checkpoints"
    checkpoint_name: str = "model-{epoch:02d}-{val_loss:.2f}"
    checkpoint_metric: str = "val/loss"
    checkpoint_mode: Literal["min", "max"] = "max"
    keep_top_k_checkpoints: int = 1
    save_last: bool = False

    # Early Stopping
    early_stop_patience: int = 200
    early_stop_min_delta: float = 1e-5

    # Project Metadata
    project_name: str = "adlr"
    run_name: Optional[str] = None
    save_dir: str = "logs"

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class ExperimentConfig:
    """Unified experiment configuration that works with any model type"""

    data: DataConfig
    model: MLPModelConfig
    training: TrainingConfig

    @classmethod
    def default_mlp(cls) -> "ExperimentConfig":
        return cls(
            data=DataConfig.random_h5(),
            model=MLPModelConfig.default(),
            training=TrainingConfig(),
        )
