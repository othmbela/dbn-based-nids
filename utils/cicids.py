import pandas as pd

import torch
from torch.utils.data.dataset import Dataset

from utils import utils


class CICIDSDataset(Dataset):

    def __init__(self, features_file, target_file, transform=None, target_transform=None):
        """
        Args:
            features_file (string): Path to the csv file with features.
            target_file (string): Path to the csv file with labels.
            transform (callable, optional): Optional transform to be applied on features.
            target_transform (callable, optional): Optional transform to be applied on labels.
        """
        self.features = pd.read_pickle(features_file)
        self.labels = pd.read_pickle(target_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx, :]
        label = self.labels.iloc[idx, :]
        if self.transform:
            feature = self.transform(feature.values)
        if self.target_transform:
            label = self.target_transform(label.values[0])
        return feature, label


def get_dataset(train_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/train",
                val_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/val",
                test_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/test"):

    # MAC Path:
    #     train_data_path=/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/train/
    #     val_data_path=/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/val/
    #     test_data_path=/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/test/

    # Windows Path:
    #     train_data_path=r"C:\Users\Othmane Belarbi\Documents\DBN-FL\datasets\train\"
    #     val_data_path=r"C:\Users\Othmane Belarbi\Documents\DBN-FL\datasets\val\"
    #     test_data_path=r"C:\Users\Othmane Belarbi\Documents\DBN-FL\datasets\test\"

    train_data = CICIDSDataset(
        features_file=f"{train_data_path}/train_features.pkl",
        target_file=f"{train_data_path}/train_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    val_data = CICIDSDataset(
        features_file=f"{val_data_path}/val_features.pkl",
        target_file=f"{val_data_path}/val_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    test_data = CICIDSDataset(
        features_file=f"{test_data_path}/test_features.pkl",
        target_file=f"{test_data_path}/test_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    return train_data, val_data, test_data


def get_balanced_dataset(train_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/train",
                         val_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/val",
                         test_data_path="/Users/othmanebelarbi/Desktop/Toshiba/DBN-FL/datasets/test"):

    train_data = CICIDSDataset(
        features_file=f"{train_data_path}/train_features_balanced.pkl",
        target_file=f"{train_data_path}/train_labels_balanced.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    val_data = CICIDSDataset(
        features_file=f"{val_data_path}/val_features.pkl",
        target_file=f"{val_data_path}/val_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    test_data = CICIDSDataset(
        features_file=f"{test_data_path}/test_features.pkl",
        target_file=f"{test_data_path}/test_labels.pkl",
        transform=torch.tensor,
        target_transform=torch.tensor
    )

    return train_data, val_data, test_data


def load_data():
    """Load training, validation and test set."""

    # Get the datasets
    train_data, val_data, test_data = get_dataset()

    samples_weight = utils.get_samples_weight(train_data.labels['label'])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight,
        len(samples_weight),
        replacement=True
    )

    batch_size = 64

    # Create the dataloaders - for training, validation and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader
