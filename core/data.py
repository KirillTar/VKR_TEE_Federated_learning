"""Data loading and Dirichlet non-IID partitioning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Literal


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD  = (0.1980, 0.2010, 0.1970)

FMNIST_MEAN = (0.2860,)
FMNIST_STD  = (0.3530,)

MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def load_dataset(
    dataset_name: Literal["cifar10", "fashion_mnist", "mnist", "svhn"],
    data_dir: str = "./data",
) -> tuple:
    """Returns (train_dataset, test_dataset)."""
    if dataset_name == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
        test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "fashion_mnist":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FMNIST_MEAN, FMNIST_STD),
        ])
        train_ds = datasets.FashionMNIST(data_dir, train=True,  download=True, transform=train_transform)
        test_ds  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)

    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ])
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "svhn":
        # SVHN: no horizontal flip (digits shouldn't be mirrored)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ])
        train_ds = datasets.SVHN(data_dir, split='train', download=True, transform=train_transform)
        test_ds  = datasets.SVHN(data_dir, split='test',  download=True, transform=test_transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, test_ds


def dirichlet_split(
    dataset,
    n_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices among clients via Dirichlet(alpha).
    alpha → ∞: IID; alpha = 0.5: moderate non-IID; alpha = 0.1: strong non-IID.
    Returns list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets if hasattr(dataset, "targets") else dataset.labels)
    n_classes = int(targets.max()) + 1
    n_samples = len(targets)

    # Group sample indices by class
    class_indices = [np.where(targets == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        rng.shuffle(class_indices[c])

    client_indices: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        # Convert to counts
        counts = (proportions * len(class_indices[c])).astype(int)
        # Fix rounding: assign remainder to random client
        diff = len(class_indices[c]) - counts.sum()
        if diff > 0:
            for _ in range(diff):
                counts[rng.integers(n_clients)] += 1

        start = 0
        for k in range(n_clients):
            end = start + counts[k]
            client_indices[k].extend(class_indices[c][start:end].tolist())
            start = end

    return client_indices


def get_client_dataloader(
    dataset,
    indices: list[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_test_dataloader(
    dataset,
    batch_size: int = 256,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_class_distribution(indices: list[int], dataset) -> dict[int, int]:
    """Return class counts for a client's data split."""
    targets = np.array(dataset.targets if hasattr(dataset, "targets") else dataset.labels)
    unique, counts = np.unique(targets[indices], return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))
