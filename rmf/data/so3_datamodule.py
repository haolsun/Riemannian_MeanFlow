import os
from torch import Tensor
import numpy as np
from lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.spatial.transform import Rotation
import torch.nn.functional as F


def load_data(dirname, filename):
    filename = os.path.join(dirname, filename)
    dataset = np.load(filename).astype("float32")
    dataset = dataset[:20000]
    dataset = torch.from_numpy(dataset).float()
    return dataset

def load_swiss():
    N = 20000
    noise = 0.1
    proj_y = 10.
    generator = torch.Generator()
    generator.manual_seed(42)
    t = 3 * np.pi * (1 - torch.rand(N, generator=generator))
    x = t * torch.cos(t)
    z = t * torch.sin(t)
    x += noise * torch.randn(N, dtype=torch.float, generator=generator)
    z += noise * torch.randn(N, dtype=torch.float, generator=generator)

    target = F.normalize(torch.stack([x, torch.ones_like(x) * proj_y, z], dim=1), dim=1)
    source = torch.tensor([0, 1, 0], dtype=torch.float).unsqueeze(0)
    axis = torch.cross(source, target, dim=1)
    theta = torch.acos(torch.clamp(torch.sum(source * target, dim=1), -1.0, 1.0))
    dataset = F.normalize(axis, dim=-1) * theta.unsqueeze(1)
    dataset = Rotation.from_rotvec(dataset).as_matrix()
    # print('dataset: ',dataset.shape)
    data = torch.Tensor(dataset).float()
    return data

class SpecialOrthogonalGroup(Dataset):
    def __init__(self, dirname, filename):
        if filename == "swiss":
            self.N = 40000
            self.noise = 0.01
            self.proj_y = 10.
            generator = torch.Generator()
            generator.manual_seed(42)
            t = 3 * np.pi * (1 - torch.rand(self.N, generator=generator))
            x = t * torch.cos(t)
            z = t * torch.sin(t)
            x += self.noise * torch.randn(self.N, dtype=torch.float, generator=generator)
            z += self.noise * torch.randn(self.N, dtype=torch.float, generator=generator)

            target = F.normalize(torch.stack([x, torch.ones_like(x) * self.proj_y, z], dim=1), dim=1)
            source = torch.tensor([0, 1, 0], dtype=torch.float).unsqueeze(0)
            axis = torch.cross(source, target, dim=1)
            theta = torch.acos(torch.clamp(torch.sum(source * target, dim=1), -1.0, 1.0))
            dataset = F.normalize(axis, dim=-1) * theta.unsqueeze(1)
            dataset = Rotation.from_rotvec(dataset).as_matrix()
            np.random.shuffle(dataset)
            # print('dataset: ', dataset.shape)
        elif filename == "so3": # cfg
            data_1 = load_data(dirname, 'cone_train.npy')
            data_2 = load_data(dirname, 'fisher24_train.npy')
            data_3 = load_swiss()
            dataset = torch.cat((data_1, data_2, data_3), dim=0)
            np.random.shuffle(dataset)
            self.data = torch.Tensor(dataset).float()
        else:
            filename_0 = os.path.join(dirname, filename + '_train.npy')
            dataset = np.load(filename_0).astype("float32")
            filename_1 = os.path.join(dirname, filename + "_test.npy")
            dataset_1 = np.load(filename_1).astype("float32")
            dataset = np.concatenate((dataset, dataset_1), axis=0)
            np.random.shuffle(dataset)
        self.data = torch.Tensor(dataset[:40000]).float()
        # print('self.data',self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Cone(SpecialOrthogonalGroup):
    def __init__(self, dirname):
        super().__init__(dirname, "cone")


class Fisher(SpecialOrthogonalGroup):
    def __init__(self, dirname):
        super().__init__(dirname, "fisher24")


class Line(SpecialOrthogonalGroup):
    def __init__(self, dirname):
        super().__init__(dirname, "line")


class Peak(SpecialOrthogonalGroup):
    def __init__(self, dirname):
        super().__init__(dirname, "peak")

class Swiss(SpecialOrthogonalGroup):
    def __init__(self, dirname):
        super().__init__(dirname, "swiss")


class SO3DataModule(LightningDataModule):
    """
    so3 datasets.
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        dataset_file: str = "cone",
        batch_size: int = 64,
        num_workers: int = 1,
        pin_memory: bool = False,
    ):
        """Initialize a `SO3DataModule`.

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
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
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
            dataset = SpecialOrthogonalGroup(
                dirname=self.hparams.data_dir,
                filename=f"{self.hparams.dataset_file}",
            )
            n_val = int(0.1 * len(dataset))
            n_test = int(0.1 * len(dataset))
            n_train = len(dataset) - n_val - n_test
            data_train, self.data_val, self.data_test = random_split(
                dataset,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = data_train

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def get_test_tensor(self) -> Tensor:
        if self.data_test is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_test[i] for i in range(len(self.data_test))])

    def get_val_tensor(self) -> Tensor:
        if self.data_val is None:
            raise RuntimeError("The test set is not available.")
        return torch.stack([self.data_val[i] for i in range(len(self.data_val))])


# do not eval MMD on validation set
if __name__ == "__main__":
    dm = SO3DataModule('../../data\\raw')
    dm.setup()
    for batch in dm.train_dataloader():
        x = dm.get_test_tensor()
        # x = x.unsqueeze(1)
        print('x:',x.shape)
        break
        # print(batch.shape)
        # import ipdb

        # ipdb.set_trace()