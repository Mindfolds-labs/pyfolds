"""Treino de classificador com MPJRDLayer + MPJRDNeuronV2.

Preferência: MNIST (torchvision).
Fallback automático: dataset sintético simples quando torchvision não está disponível.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from pyfolds import MPJRDConfig
from pyfolds.core import MPJRDNeuronV2
from pyfolds.layers import MPJRDLayer


@dataclass
class Metrics:
    loss: float
    acc: float


def build_mpjrd_feature_layer(device: torch.device, height: int, width: int) -> MPJRDLayer:
    cfg = MPJRDConfig(
        n_dendrites=height,
        n_synapses_per_dendrite=width,
        theta_init=max(3.0, float(height) * 0.35),
        target_spike_rate=0.1,
        plastic=False,
    )
    return MPJRDLayer(
        n_neurons=128,
        cfg=cfg,
        name="feature_layer",
        neuron_class=MPJRDNeuronV2,
        enable_telemetry=False,
        device=device,
    ).eval()


class SpikeReadout(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 10):
        super().__init__()
        self.head = nn.Linear(n_features, n_classes)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        return self.head(spikes)


def _to_layer_input(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def evaluate(feature_layer: MPJRDLayer, readout: SpikeReadout, loader: DataLoader, device: torch.device) -> Metrics:
    feature_layer.eval()
    readout.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_hits = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = readout(feature_layer(_to_layer_input(x))["spikes"])
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            total_hits += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return Metrics(loss=total_loss / max(total, 1), acc=total_hits / max(total, 1))


def train_readout(
    feature_layer: MPJRDLayer,
    readout: SpikeReadout,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[Metrics, Metrics]:
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        readout.train()
        total_loss = 0.0
        total_hits = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                spikes = feature_layer(_to_layer_input(x))["spikes"]

            logits = readout(spikes)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_hits += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_metrics = Metrics(loss=total_loss / total, acc=total_hits / total)
        test_metrics = evaluate(feature_layer, readout, test_loader, device)
        print(
            f"epoch={epoch+1} train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.acc:.3f} test_loss={test_metrics.loss:.4f} "
            f"test_acc={test_metrics.acc:.3f}"
        )

    return train_metrics, test_metrics


def _load_mnist_torchvision(batch_size: int, subset_train: int, subset_test: int):
    from torchvision import datasets, transforms

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if subset_train > 0:
        train_set = Subset(train_set, range(min(subset_train, len(train_set))))
    if subset_test > 0:
        test_set = Subset(test_set, range(min(subset_test, len(test_set))))

    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size), 28, 28, "mnist"


def _load_synthetic(batch_size: int, subset_train: int, subset_test: int):
    h = w = 8
    n_train = subset_train if subset_train > 0 else 2000
    n_test = subset_test if subset_test > 0 else 500

    def make_set(n: int):
        x = torch.zeros(n, 1, h, w)
        y = torch.randint(0, 10, (n,))
        for i in range(n):
            cls = int(y[i].item())
            x[i, 0, cls % h, :] = 1.0
            x[i, 0, :, (cls * 3) % w] += 0.5
            x[i] += 0.05 * torch.randn(1, h, w)
            x[i].clamp_(0.0, 1.0)
        return TensorDataset(x, y)

    train_set = make_set(n_train)
    test_set = make_set(n_test)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size), h, w, "synthetic-fallback"


def load_dataset(batch_size: int, subset_train: int, subset_test: int):
    try:
        return _load_mnist_torchvision(batch_size, subset_train, subset_test)
    except Exception as exc:
        print(f"Aviso: MNIST indisponível ({exc}). Usando dataset sintético de fallback.")
        return _load_synthetic(batch_size, subset_train, subset_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset-train", type=int, default=5000)
    parser.add_argument("--subset-test", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_loader, test_loader, h, w, dataset_name = load_dataset(args.batch_size, args.subset_train, args.subset_test)
    print(f"dataset={dataset_name} shape={h}x{w}")

    feature_layer = build_mpjrd_feature_layer(device, h, w)
    readout = SpikeReadout(n_features=feature_layer.n_neurons).to(device)

    train_metrics, test_metrics = train_readout(
        feature_layer, readout, train_loader, test_loader, device, args.epochs, args.lr
    )

    print("\nResumo final")
    print(f"train_acc={train_metrics.acc:.3f}")
    print(f"test_acc={test_metrics.acc:.3f}")
    print("Status:", "bom" if test_metrics.acc >= 0.80 else "precisa ajustar")


if __name__ == "__main__":
    main()
