"""Treino MNIST com MPJRD Wave ativado + readout linear.

Objetivo prático: validar pipeline Wave sem falhas e medir acurácia básica.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pyfolds.layers import MPJRDWaveLayer
from pyfolds.wave import MPJRDWaveConfig

logging.getLogger("pyfolds.core.dendrite").setLevel(logging.ERROR)


@dataclass
class Metrics:
    loss: float
    acc: float


class WaveReadout(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 10):
        super().__init__()
        self.head = nn.Linear(n_features * 3, n_classes)

    def forward(self, spikes: torch.Tensor, wave_real: torch.Tensor, wave_imag: torch.Tensor) -> torch.Tensor:
        features = torch.cat([spikes, wave_real, wave_imag], dim=1)
        return self.head(features)


def _to_layer_input(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def build_loaders(batch_size: int, subset_train: int, subset_test: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if subset_train > 0:
        train_ds = Subset(train_ds, range(min(subset_train, len(train_ds))))
    if subset_test > 0:
        test_ds = Subset(test_ds, range(min(subset_test, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate(layer: MPJRDWaveLayer, readout: WaveReadout, loader: DataLoader, device: torch.device) -> Metrics:
    layer.eval()
    readout.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_hits = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = layer(_to_layer_input(x), neuron_kwargs={"target_class": None})
            logits = readout(out["spikes"], out["wave_real"], out["wave_imag"])
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            total_hits += int((logits.argmax(dim=1) == y).sum().item())
            total += int(x.size(0))

    return Metrics(loss=total_loss / max(1, total), acc=total_hits / max(1, total))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-train", type=int, default=2000)
    parser.add_argument("--subset-test", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_loader, test_loader = build_loaders(args.batch_size, args.subset_train, args.subset_test)

    cfg = MPJRDWaveConfig(
        n_dendrites=28,
        n_synapses_per_dendrite=28,
        theta_init=4.0,
        plastic=False,
        wave_enabled=True,
        base_frequency=12.0,
        frequency_step=4.0,
        class_frequencies=tuple(12.0 + 4.0 * i for i in range(10)),
    )

    wave_layer = MPJRDWaveLayer(
        n_neurons=16,
        cfg=cfg,
        name="wave_feature_layer",
        enable_telemetry=False,
        device=device,
    ).eval()
    readout = WaveReadout(n_features=wave_layer.n_neurons).to(device)

    optimizer = torch.optim.Adam(readout.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        readout.train()
        total_loss = 0.0
        total_hits = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out = wave_layer(_to_layer_input(x), neuron_kwargs={"target_class": None})
                spikes = out["spikes"]
                wave_real = out["wave_real"]
                wave_imag = out["wave_imag"]

            logits = readout(spikes, wave_real, wave_imag)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            total_hits += int((logits.argmax(dim=1) == y).sum().item())
            total += int(x.size(0))

        train_metrics = Metrics(loss=total_loss / max(1, total), acc=total_hits / max(1, total))
        test_metrics = evaluate(wave_layer, readout, test_loader, device)

        print(
            f"epoch={epoch+1} train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.acc * 100:.2f}% test_loss={test_metrics.loss:.4f} "
            f"test_acc={test_metrics.acc * 100:.2f}%"
        )

    final_metrics = evaluate(wave_layer, readout, test_loader, device)
    print("\nResumo final")
    print(f"test_acc={final_metrics.acc * 100:.2f}%")
    print("status=OK" if final_metrics.acc >= 0.02 else "status=ABAIXO_DO_MINIMO")


if __name__ == "__main__":
    main()
