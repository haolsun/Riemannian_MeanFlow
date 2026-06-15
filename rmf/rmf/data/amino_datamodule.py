import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from lightning import LightningDataModule

from gfm.data.components.expand import ExpandDataset


class Top500(Dataset):
    dim = 2

    def __init__(self, root="data/top500", amino="General"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro", "top"]
        assert amino in amino_types, f"amino type {amino} not implemented"
        if amino == 'top':
            data_1 = data = data[data["amino"] == "General"][["phi", "psi"]].values.astype("float32") # general
            data_2 = data = data[data["amino"] == "General"][["phi", "psi"]].values.astype("float32") # general
            data_3 = data = data[data["amino"] == "General"][["phi", "psi"]].values.astype("float32") # general
            data_4 = data = data[data["amino"] == "General"][["phi", "psi"]].values.astype("float32") # general

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        print(data.shape)
        self.data = torch.tensor(data % 360 * torch.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RNA(Dataset):
    dim = 7

    def __init__(self, root="data/rna", amino=None):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=[
                "source",
                "base",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "chi",
            ],
        )

        data = data[
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        ].values.astype("float32")
        self.data = torch.tensor(data % 360 * torch.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Top500DataModule(LightningDataModule):
    """
    Top 500 dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/top500",
        data_type: str = "General",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `Top500DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        expand_factor = 1
        if self.hparams.data_type == "Glycine":
            expand_factor = 10
        elif self.hparams.data_type == "Proline":
            expand_factor = 18
        elif self.hparams.data_type == "Pre-Pro":
            expand_factor = 20
        elif self.hparams.data_type == "RNA":
            expand_factor = 14
        # Divide batch size by the number of devices.
        base_dataset = RNA if self.hparams.data_type == "RNA" else Top500
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = base_dataset(
                root=self.hparams.data_dir, amino=self.hparams.data_type
            )
            n_val = int(0.1 * len(dataset))
            n_test = int(0.1 * len(dataset))
            n_train = len(dataset) - n_val - n_test
            data_train, self.data_val, self.data_test = random_split(
                dataset,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = ExpandDataset(data_train, expand_factor=expand_factor)

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def get_test_tensor(self) -> torch.Tensor:
        """
        Returns the test data as a Tensor.
        """
        if self.data_test is None:
            raise ValueError("DataModule not setup yet.")
        return torch.stack([self.data_test[i] for i in range(len(self.data_test))])

    def get_val_tensor(self) -> torch.Tensor:
        """
        Returns the test data as a Tensor.
        """
        if self.data_val is None:
            raise ValueError("DataModule not setup yet.")
        return torch.stack([self.data_val[i] for i in range(len(self.data_val))])


if __name__ == "__main__":
    dm = Top500DataModule()
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(dm.get_test_tensor().shape)
        break
