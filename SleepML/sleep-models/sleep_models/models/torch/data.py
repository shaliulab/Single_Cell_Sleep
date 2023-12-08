import torch
from torch.utils.data import Dataset


class SingleCellDataSet(Dataset):
    def __init__(self, X, y, device=None):
        self._X = X
        self._labels = y
        self._device = device

    def __len__(self):
        return self._labels.shape[0]

    def __getitem__(self, idx):
        X = self._X[idx, :]
        label = self._labels[idx, :]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            label = self.target_transform(label)
        return X, label

    def transform(self, X):
        X = torch.from_numpy(X).type(torch.float)

        if self._device:
            X = X.to(self._device)

        return X

    def target_transform(self, y):
        y = torch.from_numpy(y).type(torch.float)

        if self._device:
            y = y.to(self._device)

        return y
