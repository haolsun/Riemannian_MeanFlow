import os
from csv import reader
import numpy as np
from lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from geomstats.geometry.hypersphere import Hypersphere
from gfm.data.components.expand import ExpandDataset
from gfm.data.components.util import cartesian_from_latlon
from gfm.manifold.manifold import FlatTorus

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = np.array(list(lines)[1:]).astype(np.float64)
    # print(dataset.shape)
    return dataset

def load_and_trans(dirname, filename, label):
    filename = os.path.join(dirname, filename)
    subdata = load_csv(filename)
    subdata = torch.Tensor(subdata)
    subdata = cartesian_from_latlon(subdata / 180 * torch.pi)
    subdata = torch.cat([subdata, torch.full([subdata.shape[0],1], label)], dim=1)
    return subdata

class EarthData(Dataset):
    dim = 3

    def __init__(self, dirname, filename):
        if filename == 'earth.csv':
            data_1 = load_and_trans(dirname, 'volcano.csv',1)
            data_1 = torch.cat([data_1 for i in range(15)], dim=0)
            data_2 = load_and_trans(dirname, 'earthquake.csv',2)
            data_2 = torch.cat([data_2 for i in range(2)], dim=0)
            data_3 = load_and_trans(dirname, 'flood.csv',3) 
            data_3 = torch.cat([data_3 for i in range(3)], dim=0)
            data_4 = load_and_trans(dirname, 'fire.csv',4)
            dataset = torch.cat([data_1, data_2, data_3, data_4], dim=0)
            dataset = dataset[torch.randperm(dataset.size(0))]
            self.data = dataset
        if filename == 'hyper.csv':
            # 维度可调
            torus = Hypersphere(127)
            data = torus.random_point(n_samples=50000)
            # print(data)
            data_1 = list()
            for i in data:
                # print(i)
                if i[1] > -0.1 and i[1] < 0.1:
                    data_1.append([j for j in i])
            data = np.array(data_1)
            dataset = torch.Tensor(data)
            self.data = dataset
        else:
            filename = os.path.join(dirname, filename)
            dataset = load_csv(filename)
            dataset = torch.Tensor(dataset)
            self.latlon = dataset
            self.data = cartesian_from_latlon(dataset / 180 * torch.pi)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Volcano(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "volcano.csv")


class Earthquake(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earthquake.csv")


class Fire(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "fire.csv")


class Flood(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "flood.csv")

class Earth(EarthData):
    def __init__(self, dirname):
        super().__init__(dirname, "earth.csv")

class EarthDataModule(LightningDataModule):
    """
    Earth datasets.
    """

    def __init__(
        self,
        data_dir: str = "data/earth_data",
        dataset_file: str = "fire",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """Initialize a `EarthDataModule`.

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
        if self.hparams.dataset_file == "volcano":
            expand_factor = 1550
        elif self.hparams.dataset_file == "earthquake":
            expand_factor = 210
        elif self.hparams.dataset_file == "fire":
            expand_factor = 100
        elif self.hparams.dataset_file == "flood":
            expand_factor = 260
        elif self.hparams.dataset_file == "earth":
            expand_factor = 1
        # Divide batch size by the number of devices.
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
            dataset = EarthData(
                dirname=self.hparams.data_dir,
                filename=f"{self.hparams.dataset_file}.csv",
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
        Get the full test set as a tensor.
        """
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i] for i in range(len(self.data_test))])

    def get_val_tensor(self) -> torch.Tensor:
        """
        Get the full val set as a tensor.
        """
        if self.data_val is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_val[i] for i in range(len(self.data_val))])

    def get_volcano_tensor(self) -> torch.Tensor:
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i][0:3] for i in range(len(self.data_test)) if self.data_test[i][3]==1])

    def get_earthquake_tensor(self) -> torch.Tensor:
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i][0:3] for i in range(len(self.data_test)) if self.data_test[i][3]==2])

    def get_flood_tensor(self) -> torch.Tensor:
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i][0:3] for i in range(len(self.data_test)) if self.data_test[i][3]==3])

    def get_fire_tensor(self) -> torch.Tensor:
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i][0:3] for i in range(len(self.data_test)) if self.data_test[i][3]==4])


# do not eval MMD on validation set
if __name__ == "__main__":
    dm = EarthDataModule(dataset_file='earth',batch_size=512)
    dm.setup()
    # print(dm.get_test_tensor().shape)
    print(dm.data_test)
    print(dm.data_val)
    print(dm.data_train)
    for batch in dm.train_dataloader():
        x = batch
        print('x:',x)
        data, label = torch.split(x, [3,1],dim=1)
        print('data:',data.shape)
        print('label:', label.shape)
        # print(label)
        print('test:',dm.get_test_tensor())
        print('volcano:', dm.get_volcano_tensor().shape)
        print('earthquake:', dm.get_earthquake_tensor().shape)
        print('flood:', dm.get_flood_tensor().shape)
        print('fire:', dm.get_fire_tensor().shape)
        break
        # print(batch.shape)
        # import ipdb

        # ipdb.set_trace()