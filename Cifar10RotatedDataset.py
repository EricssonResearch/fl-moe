from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np


class Cifar10RotatedDataset(datasets.CIFAR10):
    """
    Rotated CIFAR-10 based on A. Ghosh, D. Yin, J. Chung, and K. Ramchandran, “An Efficient Framework for Clustered Federated Learning,” arXiv, no. NeurIPS, 2020.
    """

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_clients: int = 10,
            n_data: int = -1
    ):
        super().__init__(root, train, transform, target_transform, download)

        self.width = 28
        self.height = 28

        # k=2, half of the clients select the normal data, half the rotated
        if n_data == -1:
            n_data = 2*len(self.data)//num_clients

        # Duplicate data and rotate it
        self.rot_data = self.data.copy()

        for n, img in enumerate(self.rot_data):
            self.rot_data[n] = np.rot90(img, 2)

        self.dict_users = {}

        two_split = np.array_split(range(num_clients), 2)

        all_idxs = list(range(len(self.data)))

        for i in two_split[0]:
            self.dict_users[i] = set(np.random.choice(
                all_idxs, int(n_data), replace=False))

            all_idxs = list(set(all_idxs) - self.dict_users[i])
            self.dict_users[i] = list(self.dict_users[i])

        all_idxs = list(range(len(self.data),2*len(self.data)))

        for i in two_split[1]:
            self.dict_users[i] = set(np.random.choice(
                all_idxs, int(n_data), replace=False))

            all_idxs = list(set(all_idxs) - self.dict_users[i])
            self.dict_users[i] = list(self.dict_users[i])

        self.data = np.vstack((self.data, self.rot_data))
        self.targets = self.targets + self.targets

