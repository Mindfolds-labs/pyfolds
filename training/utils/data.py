from __future__ import annotations

import importlib.util

import torch
from torch.utils.data import DataLoader, TensorDataset

if importlib.util.find_spec("torchvision") is not None:
    import torchvision
    import torchvision.transforms as transforms
else:
    torchvision = None
    transforms = None


def build_mnist_loaders(batch_size: int):
    if torchvision is not None and transforms is not None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        try:
            train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(test_ds, batch_size=1000, shuffle=False),
            )
        except Exception:
            pass

    x_train = torch.rand(2048, 1, 28, 28)
    y_train = torch.randint(0, 10, (2048,))
    x_test = torch.rand(512, 1, 28, 28)
    y_test = torch.randint(0, 10, (512,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=512, shuffle=False),
    )
