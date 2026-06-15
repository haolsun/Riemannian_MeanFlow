from torch.utils.data import Dataset


class ExpandDataset(Dataset):
    def __init__(self, dset, expand_factor=1):
        self.dset = dset
        self.expand_factor = expand_factor

    def __len__(self):
        return len(self.dset) * self.expand_factor

    def __getitem__(self, idx):
        return self.dset[idx % len(self.dset)]
