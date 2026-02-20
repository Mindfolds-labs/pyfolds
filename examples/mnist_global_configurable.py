"""Treino MNIST com projeção global para AdaptiveNeuronLayer usando configuração em arquivo.

Objetivo:
- manter o layout do exemplo fornecido (projeção 784 -> [N,D,S])
- ativar lógica de aprendizado via arquivo de configuração
- inicializar variáveis com defaults próximos à literatura (artigos)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pyfolds import LearningMode, NeuronConfig, AdaptiveNeuronLayer

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover
    torchvision = None
    transforms = None


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mode_from_config(value: str, plastic: bool) -> LearningMode:
    if not plastic:
        return LearningMode.INFERENCE
    return LearningMode(value.strip().lower())


class MNISTMPJRDGlobal(nn.Module):
    def __init__(self, cfg: NeuronConfig, n_neurons: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.device = device

        total_synapses = n_neurons * cfg.n_dendrites * cfg.n_synapses_per_dendrite
        self.proj = nn.Linear(784, total_synapses)
        self.mpjrd = AdaptiveNeuronLayer(n_neurons=n_neurons, cfg=cfg, name="global_layer", device=device)
        self.fc = nn.Linear(n_neurons, 10)

    def forward(self, x: torch.Tensor, mode: LearningMode) -> torch.Tensor:
        batch = x.size(0)
        flat = x.view(batch, -1)
        x_proj = self.proj(flat)
        x_input = x_proj.view(
            batch,
            self.n_neurons,
            self.cfg.n_dendrites,
            self.cfg.n_synapses_per_dendrite,
        )
        out = self.mpjrd(x_input, mode=mode)
        spikes = out["spikes"]
        return self.fc(spikes)


def _build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    if torchvision is None or transforms is None:
        raise RuntimeError("torchvision indisponível para carregar MNIST")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/mnist_global_config.json"),
        help="Caminho do arquivo JSON com parâmetros do experimento.",
    )
    args = parser.parse_args()

    raw = _load_config(args.config)
    training = raw["training"]
    model_cfg = raw["model"]
    learning = raw["learning"]
    scientific = raw["scientific_init"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(training["save_dir"], exist_ok=True)

    mpjrd_cfg = NeuronConfig(
        n_dendrites=model_cfg["n_dendrites"],
        n_synapses_per_dendrite=model_cfg["n_synapses_per_dendrite"],
        dendrite_integration_mode=model_cfg["dendrite_integration_mode"],
        plastic=learning["plastic"],
        defer_updates=learning["defer_updates"],
        **scientific,
    )

    mode = _mode_from_config(learning["mode"], plastic=learning["plastic"])

    print(f"Dispositivo: {device}")
    print(f"Iniciando treinamento - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Modo de aprendizado: {mode.value} | plastic={mpjrd_cfg.plastic}")

    model = MNISTMPJRDGlobal(mpjrd_cfg, n_neurons=training["n_neurons"], device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=training["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = _build_dataloaders(training["batch_size"])

    best_acc = 0.0
    for epoch in range(training["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mode=mode)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += int((pred == labels).sum().item())

        train_acc = 100.0 * correct / max(total, 1)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, mode=LearningMode.INFERENCE)
                pred = outputs.argmax(dim=1)
                test_total += labels.size(0)
                test_correct += int((pred == labels).sum().item())

        test_acc = 100.0 * test_correct / max(test_total, 1)
        print(
            f"Epoch {epoch+1:2d}/{training['epochs']} | "
            f"Loss: {running_loss / max(1, len(train_loader)):.4f} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%"
        )

        layer_metrics = model.mpjrd.get_layer_metrics()
        print(
            "  → "
            f"N médio: {layer_metrics['n_mean']:.2f} | "
            f"I médio: {layer_metrics['i_mean']:.3f} | "
            f"θ médio: {layer_metrics['theta_mean']:.3f} | "
            f"Taxa: {layer_metrics['r_hat_mean']:.3f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            filename = Path(training["save_dir"]) / f"best_{test_acc:.2f}.pth"
            torch.save(model.state_dict(), filename)
            print(f"Modelo salvo: {filename.name}")

    print(f"\nFinalizado! Melhor teste: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
