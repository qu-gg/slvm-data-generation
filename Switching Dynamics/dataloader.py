"""
@file dataloader.py
@author Ryan Missel

Holds the WebDataset classes for the available datasets
"""
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class SwitchingDataset(Dataset):
    """
    Load sequences of signals
    """

    def __init__(self, file_path):
        # Load data
        npzfile = np.load(file_path)
        self.signals = npzfile['queries'].astype(np.float32)
        print(f"=> Signals: {self.signals.shape}")

        self.domains = npzfile['domains'].astype(np.float32)
        print(f"=> Domains: {self.domains.shape}")

        # Get labels of environments
        self.labels = npzfile['labels'].astype(np.int16)

        # Get data dimensions
        self.sequences, self.timesteps, self.dim = self.signals.shape

        # Convert to Tensors
        self.signals = torch.from_numpy(self.signals)
        self.domains = torch.from_numpy(self.domains)
        self.labels = torch.from_numpy(self.labels)

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        """ Get signal and support set """
        return torch.Tensor([idx]), self.signals[idx], self.domains[idx], self.labels[idx]


if __name__ == '__main__':
    dataset = SwitchingDataset('../data/double_pendulum/double_pendulum_train.npz')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    import matplotlib.pyplot as plt
    for batch in dataloader:
        indices, signals, domains, labels = batch
        plt.plot(domains[0, 0, :, 0], range(7), marker='x')
        plt.plot(domains[0, 1, :, 0], range(7, 14), marker='x')
        plt.plot(domains[0, 2, :, 0], range(14, 21), marker='x')
        plt.plot(signals[0, :, 0], range(21, 28), marker='x')
        plt.legend()
        plt.show()
        break