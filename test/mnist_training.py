"""Treino integrado MNIST para validação arquitetural do PyFolds.

Este script segue o padrão do projeto:
- imports absolutos via pacote `pyfolds`
- logging apenas em arquivo (terminal silencioso)
- fallback para dataset sintético quando MNIST não estiver disponível
- checkpoint versionado via `VersionedCheckpoint`

Execução sugerida:
    PYTHONPATH=src python test/mnist_training.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from pyfolds import LearningMode, MPJRDConfig, MPJRDLayer, VersionedCheckpoint, __version__
from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.utils.logging import setup_run_logging

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover
    datasets = None
    transforms = None


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 1
    learning_rate: float = 1e-3
    n_neurons: int = 24
    train_limit: int = 512
    test_limit: int = 128


class SyntheticMNIST(Dataset):
    """Dataset sintético no mesmo formato do MNIST para ambiente offline."""

    def __init__(self, size: int, seed: int = 42) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 1, 28, 28, generator=gen)
        self.labels = torch.randint(0, 10, (size,), generator=gen)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


class MnistMPJRDModel(nn.Module):
    """Modelo de classificação com camada MPJRD integrada ao fluxo PyTorch."""

    def __init__(self, cfg: MPJRDConfig, n_neurons: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.total_synapses = n_neurons * cfg.n_dendrites * cfg.n_synapses_per_dendrite

        self.encoder = nn.Linear(784, self.total_synapses)
        self.hidden = MPJRDLayer(
            n_neurons=n_neurons,
            cfg=cfg,
            neuron_cls=MPJRDNeuronAdvanced,
            enable_telemetry=False,
            device=device,
            name="mnist_hidden",
        )
        self.head = nn.Linear(n_neurons, 10)

    def forward(self, x: torch.Tensor, mode: LearningMode) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        encoded = torch.sigmoid(self.encoder(x.view(batch, -1))).view(
            batch, self.n_neurons, self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite
        )
        out = self.hidden(encoded, mode=mode)
        spikes = out["spikes"].float()
        logits = self.head(spikes)
        return logits, spikes


def _subset(ds: Dataset, limit: int) -> Dataset:
    return torch.utils.data.Subset(ds, range(min(limit, len(ds))))


def _load_mnist(cfg: TrainingConfig, logger) -> tuple[DataLoader, DataLoader, str]:
    if datasets is None or transforms is None:
        logger.warning("torchvision indisponível: usando dataset sintético")
        train_ds = SyntheticMNIST(cfg.train_limit)
        test_ds = SyntheticMNIST(cfg.test_limit, seed=99)
        source = "synthetic"
    else:
        try:
            tfm = transforms.Compose([transforms.ToTensor()])
            train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
            test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
            train_ds = _subset(train_ds, cfg.train_limit)
            test_ds = _subset(test_ds, cfg.test_limit)
            source = "mnist"
        except Exception as exc:  # pragma: no cover
            logger.warning("Falha ao carregar MNIST (%s): usando sintético", exc)
            train_ds = SyntheticMNIST(cfg.train_limit)
            test_ds = SyntheticMNIST(cfg.test_limit, seed=99)
            source = "synthetic"

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, source


def _evaluate(model: MnistMPJRDModel, loader: Iterable, device: torch.device) -> float:
    model.eval()
    model.hidden.set_mode(LearningMode.INFERENCE)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, mode=LearningMode.INFERENCE)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.shape[0])
    return 0.0 if total == 0 else (100.0 * correct / total)


def run_training(cfg: TrainingConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrainingConfig()
    logger, log_path = setup_run_logging(app="mnist_training", console=False, level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mp_cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8, plastic=True)
    model = MnistMPJRDModel(mp_cfg, n_neurons=cfg.n_neurons, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader, source = _load_mnist(cfg, logger)
    logger.info("source=%s train_batches=%d test_batches=%d", source, len(train_loader), len(test_loader))

    for epoch in range(cfg.epochs):
        model.train()
        model.hidden.set_mode(LearningMode.ONLINE)
        epoch_loss = 0.0
        epoch_spike = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, spikes = model(images, mode=LearningMode.ONLINE)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_spike += float(spikes.mean().item())
        logger.info(
            "epoch=%d loss=%.4f spike_rate=%.4f",
            epoch + 1,
            epoch_loss / max(1, len(train_loader)),
            epoch_spike / max(1, len(train_loader)),
        )

    acc = _evaluate(model, test_loader, device)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    ckpt = output_dir / "mnist_training_checkpoint.pt"
    VersionedCheckpoint(model=model, version=__version__).save(
        str(ckpt), extra_metadata={"source": source, "accuracy": acc}
    )
    logger.info("checkpoint=%s accuracy=%.2f", ckpt, acc)

    return {
        "accuracy": acc,
        "dataset": source,
        "log_path": str(log_path),
        "checkpoint": str(ckpt),
    }


if __name__ == "__main__":
    run_training()
