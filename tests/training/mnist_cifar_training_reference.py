"""Referência de treinamento MNIST/CIFAR usando imports padrão de ``pyfolds``.

Arquivo intencionalmente mantido em ``tests/training`` para análise posterior,
sem execução automática na suíte de testes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyfolds import MPJRDConfig, MPJRDNeuron


@dataclass
class TrainingConfig:
    dataset: str = "mnist"  # "mnist" | "cifar10"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 2
    hidden_dim: int = 256
    data_root: str = "./.data"


class PyFoldsMLP(nn.Module):
    """MLP simples com gate de ativação baseado em MPJRDNeuron."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Exemplo de integração: objeto pyfolds para controle/inspeção do gate.
        self.gate_neuron = MPJRDNeuron(MPJRDConfig(n_dendrites=4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))

        # Gate simples: usa energia média como entrada e mantém contrato pyfolds.
        mean_energy = float(x.detach().mean().item())
        gate = 1.0 if self.gate_neuron.forward(mean_energy) > 0 else 0.5
        x = x * gate

        return self.fc2(x)


def build_dataset(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader, int]:
    if cfg.dataset.lower() == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_ds = datasets.MNIST(cfg.data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(cfg.data_root, train=False, download=True, transform=transform)
        input_dim = 28 * 28
    elif cfg.dataset.lower() == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_ds = datasets.CIFAR10(
            cfg.data_root, train=True, download=True, transform=transform
        )
        test_ds = datasets.CIFAR10(
            cfg.data_root, train=False, download=True, transform=transform
        )
        input_dim = 32 * 32 * 3
    else:
        raise ValueError("dataset deve ser 'mnist' ou 'cifar10'")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, input_dim


def train_reference(cfg: TrainingConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, input_dim = build_dataset(cfg)

    model = PyFoldsMLP(input_dim, cfg.hidden_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        acc = evaluate(model, test_loader, device)
        print(f"epoch={epoch+1} loss={avg_loss:.4f} test_acc={acc:.2f}%")


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.size(0))
    return 100.0 * correct / max(1, total)


if __name__ == "__main__":
    # Exemplos:
    #   python tests/training/mnist_cifar_training_reference.py
    #   python tests/training/mnist_cifar_training_reference.py --dataset cifar10
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    args = parser.parse_args()

    config = TrainingConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    train_reference(config)
