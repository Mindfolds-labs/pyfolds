from __future__ import annotations

import importlib.util
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

if importlib.util.find_spec("torchvision") is not None:
    import torchvision
    import torchvision.transforms as transforms
else:
    torchvision = None
    transforms = None


_DATASET_META = {
    "mnist": {"channels": 1, "size": 28, "classes": 10},
    "cifar10": {"channels": 3, "size": 32, "classes": 10},
    "cifar100": {"channels": 3, "size": 32, "classes": 100},
}


def _synthetic_loaders(dataset: str, batch_size: int):
    meta = _DATASET_META[dataset]
    train_samples = int(os.getenv("PYFOLDS_SYNTH_TRAIN_SAMPLES", "2048"))
    test_samples = int(os.getenv("PYFOLDS_SYNTH_TEST_SAMPLES", "512"))
    x_train = torch.rand(train_samples, meta["channels"], meta["size"], meta["size"])
    y_train = torch.randint(0, meta["classes"], (train_samples,))
    x_test = torch.rand(test_samples, meta["channels"], meta["size"], meta["size"])
    y_test = torch.randint(0, meta["classes"], (test_samples,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=min(1000, test_samples), shuffle=False),
    )


def build_image_loaders(dataset: str, batch_size: int):
    key = dataset.lower()
    if key not in _DATASET_META:
        raise ValueError(f"dataset inválido: {dataset}")

    force_synthetic = os.getenv("PYFOLDS_MNIST_SYNTHETIC", "0") == "1"
    if force_synthetic:
        return _synthetic_loaders(key, batch_size)

    if torchvision is not None and transforms is not None:
        try:
            if key == "mnist":
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
                test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            elif key == "cifar10":
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
                )
                train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
                test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
                )
                train_ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
                test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(test_ds, batch_size=1000, shuffle=False),
            )
        except Exception:
            pass

    return _synthetic_loaders(key, batch_size)


def build_mnist_loaders(batch_size: int):
    return build_image_loaders("mnist", batch_size)
