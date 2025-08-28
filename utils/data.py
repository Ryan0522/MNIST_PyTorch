import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple

def get_dataloaders(
    data_root: str = "data",
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return train/test DataLoaders for MNIST.
    """
    tx = transforms.ToTensor()

    train_ds = datasets.MNIST(root=data_root, train=True, transform=tx, download=True)
    test_ds  = datasets.MNIST(root=data_root, train=False, transform=tx, download=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
