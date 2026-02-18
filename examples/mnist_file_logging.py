"""Treino MNIST com PyFolds com logs somente em arquivo.

Inclui fallback para dataset sintético quando o download do MNIST falha,
para garantir execução completa mesmo em ambientes sem rede.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pyfolds
from pyfolds import LearningMode, MPJRDConfig, MPJRDLayer, VersionedCheckpoint
from pyfolds.advanced import MPJRDNeuronAdvanced

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover - fallback path covered by synthetic dataset
    torchvision = None
    transforms = None


def _reset_root_logging() -> None:
    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logging.root.setLevel(logging.WARNING)


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 1
    lr: float = 1e-3
    n_neurons: int = 24
    train_limit: int = 512
    test_limit: int = 128


class SyntheticMNISTDataset(Dataset):
    """Dataset sintético com o mesmo formato do MNIST."""

    def __init__(self, size: int, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(size, 1, 28, 28, generator=gen)
        self.labels = torch.randint(0, 10, (size,), generator=gen)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


class MNISTMPJRDModel(nn.Module):
    def __init__(self, cfg: MPJRDConfig, n_neurons: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.device = device
        self.total_synapses = n_neurons * cfg.n_dendrites * cfg.n_synapses_per_dendrite

        self.proj = nn.Linear(784, self.total_synapses)
        nn.init.uniform_(self.proj.weight, -0.2, 0.2)

        self.mpjrd_layer = MPJRDLayer(
            n_neurons=n_neurons,
            cfg=cfg,
            name="hidden",
            neuron_cls=MPJRDNeuronAdvanced,
            enable_telemetry=False,
            device=device,
        )
        self.classifier = nn.Linear(n_neurons, 10)

    def forward(
        self, x: torch.Tensor, mode: LearningMode
    ) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
        batch = x.size(0)
        flat = x.view(batch, -1)
        proj = self.proj(flat)
        reshaped = proj.view(
            batch, self.n_neurons, self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite
        )
        out = self.mpjrd_layer(reshaped, mode=mode)
        spikes = out["spikes"]
        logits = self.classifier(spikes)
        return logits, float(spikes.float().mean().item()), out


def _setup_file_logger() -> tuple[logging.Logger, Path]:
    _reset_root_logging()
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"pyfolds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("PYFOLDS")
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = False

    for lib in ["matplotlib", "PIL", "torch", "urllib3", "requests"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
        logging.getLogger(lib).propagate = False

    for existing_name in list(logging.root.manager.loggerDict.keys()):
        if existing_name.startswith("pyfolds"):
            lib_logger = logging.getLogger(existing_name)
            lib_logger.handlers = [file_handler]
            lib_logger.setLevel(logging.INFO)
            lib_logger.propagate = False

    return logger, log_path


def _subset(dataset: Dataset, limit: int) -> Dataset:
    limit = min(limit, len(dataset))
    return torch.utils.data.Subset(dataset, range(limit))


def build_dataloaders(cfg: TrainConfig, logger: logging.Logger) -> tuple[DataLoader, DataLoader, str]:
    if torchvision is not None and transforms is not None:
        transform = transforms.Compose([transforms.ToTensor()])
        try:
            train_ds = torchvision.datasets.MNIST(
                "./data", train=True, download=True, transform=transform
            )
            test_ds = torchvision.datasets.MNIST(
                "./data", train=False, download=True, transform=transform
            )
            logger.info("MNIST carregado com sucesso")
            source = "mnist"
        except Exception as exc:
            logger.warning("Falha no download do MNIST: %s. Usando dataset sintético.", exc)
            train_ds = SyntheticMNISTDataset(cfg.train_limit)
            test_ds = SyntheticMNISTDataset(cfg.test_limit, seed=99)
            source = "synthetic"
    else:
        logger.warning("torchvision indisponível. Usando dataset sintético.")
        train_ds = SyntheticMNISTDataset(cfg.train_limit)
        test_ds = SyntheticMNISTDataset(cfg.test_limit, seed=99)
        source = "synthetic"

    if source == "mnist":
        train_ds = _subset(train_ds, cfg.train_limit)
        test_ds = _subset(test_ds, cfg.test_limit)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    logger.info("Dataloader pronto train=%d test=%d source=%s", len(train_ds), len(test_ds), source)
    return train_loader, test_loader, source


def evaluate(model: MNISTMPJRDModel, loader: Iterable, device: torch.device) -> tuple[float, float]:
    model.eval()
    model.mpjrd_layer.set_mode(LearningMode.INFERENCE)
    correct = 0
    total = 0
    rates: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, spike_rate, _ = model(images, mode=LearningMode.INFERENCE)
            pred = logits.argmax(dim=1)
            total += labels.size(0)
            correct += int((pred == labels).sum().item())
            rates.append(spike_rate)

    if total == 0:
        return 0.0, 0.0
    return 100.0 * correct / total, float(sum(rates) / max(1, len(rates)))


def save_versioned_checkpoint(model: nn.Module, output_dir: str = "outputs") -> str:
    """Salva checkpoint versionado compatível com ecossistema PyFolds."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "mnist_file_logging_checkpoint.pt"
    checkpoint = VersionedCheckpoint(model=model, version=getattr(pyfolds, "__version__", "unknown"))
    checkpoint.save(str(ckpt_path), extra_metadata={"example": "mnist_file_logging"})
    return str(ckpt_path)


def run_training(config_override: TrainConfig | None = None) -> dict[str, object]:
    cfg = config_override or TrainConfig()
    logger, log_path = _setup_file_logger()

    def print_status(msg: str) -> None:
        print(f"▶ {msg}")
        logger.info(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_status(f"Iniciando - PyTorch {torch.__version__} / pyfolds {pyfolds.__version__}")
    print_status(f"Device: {device}")

    mp_cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        n_min=1,
        n_max=31,
        w_scale=2.0,
        i_eta=0.5,
        i_gamma=0.9,
        i_ltp_th=1.5,
        i_ltd_th=-1.5,
        theta_init=1.0,
        theta_min=0.3,
        theta_max=2.5,
        target_spike_rate=0.3,
        homeostasis_eta=0.2,
        plasticity_mode="both",
        tau_pre=20.0,
        tau_post=20.0,
        A_plus=0.02,
        A_minus=0.02,
        device="auto",
        random_seed=42,
    )

    train_loader, test_loader, source = build_dataloaders(cfg, logger)
    print_status(f"Dataset={source} train={len(train_loader.dataset)} test={len(test_loader.dataset)}")

    model = MNISTMPJRDModel(mp_cfg, n_neurons=cfg.n_neurons, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        model.mpjrd_layer.set_mode(LearningMode.ONLINE)
        epoch_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            logits, _, _ = model(images, mode=LearningMode.ONLINE)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            if batch_idx % 20 == 0:
                logger.info("Epoch %d Batch %d loss=%.4f", epoch, batch_idx, float(loss.item()))

        acc, rate = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print_status(
            f"Época {epoch}/{cfg.epochs} loss={epoch_loss / max(1, len(train_loader)):.4f} "
            f"test_acc={acc:.2f}% rate={rate:.4f}"
        )

    elapsed = time.time() - start
    checkpoint_path = save_versioned_checkpoint(model)
    logger.info("Treino finalizado tempo=%.2fs best_acc=%.2f ckpt=%s", elapsed, best_acc, checkpoint_path)
    print_status(f"✅ Treino concluído em {elapsed:.1f}s")
    print_status(f"💾 Checkpoint salvo em: {checkpoint_path}")
    print_status(f"📝 Log salvo em: {log_path}")

    return {
        "best_acc": best_acc,
        "log_path": str(log_path),
        "dataset_source": source,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
    }


if __name__ == "__main__":
    torch.set_num_threads(4)
    try:
        result = run_training()
        sys.exit(0 if result["best_acc"] >= 0.0 else 1)
    except Exception:  # pragma: no cover - script path
        logging.getLogger("PYFOLDS").exception("❌ ERRO DURANTE EXECUÇÃO")
        sys.exit(1)
