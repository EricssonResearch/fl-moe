"""
FEMIST dataset from LEAF
"""

import json
import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FemnistDataset(Dataset):
    """FEMNIST dataset."""

    def __init__(self, root_dir, train=True, transform=None, random_seed=42):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.width = 28
        self.data = torch.tensor([])
        self.targets = torch.LongTensor([])

        self.clients = []
        self.groups = []
        self.dict_users = {}

        # Select training set or test set
        dataset = "train"

        files = os.listdir(os.path.join(root_dir, dataset))
        files = [f for f in files if f.endswith('.json')]

        for f in files:

            with open(os.path.join(root_dir, dataset, f), 'r') as inf:
                cdata = json.load(inf)

            # List of clients
            self.clients.extend(cdata['users'])

            for user, data in cdata['user_data'].items():

                # Figure out the index of this data in the dataset
                start_index = len(self.data)
                end_index = start_index + len(data['x'])
                idx = list(range(start_index, end_index))

                X = torch.reshape(torch.tensor(data['x']),
                                  (-1, self.width, self.width))

                y = torch.LongTensor(data['y'])

                train_idx, val_idx = self.train_val_dataset(
                    list(range(len(idx))), random_state=random_seed)

                # Extend data tensor
                self.data = torch.cat(
                    (self.data, X))

                # Extend the target tensor
                self.targets = torch.cat(
                    (self.targets,
                     y))

                if train:
                    selected_idx = torch.tensor(idx)[train_idx].tolist()
                else:
                    selected_idx = torch.tensor(idx)[val_idx].tolist()

                # Check if this user already exists in the dictionary
                if user in self.dict_users:
                    self.dict_users[user].extend(selected_idx)
                else:
                    self.dict_users[user] = selected_idx

        self.root_dir = root_dir
        self.transform = transform

    def train_val_dataset(self, X, val_split=0.2, random_state=42):
        """
        Split dataset into train and test sets.
        """

        train_idx, val_idx = train_test_split(
            X, test_size=val_split, random_state=random_state)

        return train_idx, val_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]
        target = self.targets[idx]

        sample = (
            torch.reshape(image, (1, self.width, self.width)),
            target)

        if self.transform:
            sample = self.transform(sample)

        return sample
