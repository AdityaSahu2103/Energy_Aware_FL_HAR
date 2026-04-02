"""
Data loader for UCI HAR Dataset.
Loads features, labels, and subject IDs, then partitions by subject for FL.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from src.config import DATASET_PATH, BATCH_SIZE, NUM_FEATURES


class HARDataset(Dataset):
    """PyTorch Dataset for HAR data."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_raw_data(split="train"):
    """Load raw UCI HAR data for a given split (train/test)."""
    split_dir = os.path.join(DATASET_PATH, split)

    X = np.loadtxt(os.path.join(split_dir, f"X_{split}.txt"))
    y = np.loadtxt(os.path.join(split_dir, f"y_{split}.txt"), dtype=int)
    subjects = np.loadtxt(os.path.join(split_dir, f"subject_{split}.txt"), dtype=int)

    # Labels are 1-indexed → convert to 0-indexed
    y = y - 1

    return X, y, subjects


def load_and_partition_data():
    """
    Load UCI HAR data and partition by subject for Federated Learning.

    Returns:
        client_data: dict {subject_id: (X_train, y_train)} — one per client
        test_dataset: HARDataset for global evaluation
        scaler: fitted StandardScaler
    """
    # Load train and test splits
    X_train, y_train, subjects_train = load_raw_data("train")
    X_test, y_test, _ = load_raw_data("test")

    # Standardize features using training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Partition training data by subject → each subject = 1 FL client
    unique_subjects = np.unique(subjects_train)
    client_data = {}

    for subject_id in unique_subjects:
        mask = subjects_train == subject_id
        client_data[subject_id] = (X_train[mask], y_train[mask])

    # Global test set
    test_dataset = HARDataset(X_test, y_test)

    print(f"[Data] Loaded UCI HAR Dataset")
    print(f"  ├── Training samples: {len(X_train)} across {len(unique_subjects)} subjects")
    print(f"  ├── Test samples: {len(X_test)}")
    print(f"  └── Features: {NUM_FEATURES}")
    print(f"  Client data distribution:")
    for sid in sorted(client_data.keys()):
        X, y = client_data[sid]
        activities = np.bincount(y, minlength=6)
        print(f"      Client {sid:2d}: {len(X):4d} samples | Activities: {activities}")

    return client_data, test_dataset, scaler


def get_client_dataloader(features, labels, batch_size=BATCH_SIZE, shuffle=True):
    """Create a DataLoader from numpy arrays."""
    dataset = HARDataset(features, labels)
    # drop_last=True prevents batch_size=1 which crashes BatchNorm
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=(len(dataset) > batch_size))


def get_test_dataloader(test_dataset, batch_size=BATCH_SIZE):
    """Create a DataLoader for the test dataset."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
