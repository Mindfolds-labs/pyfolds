"""Treino e inferência MNIST para validação arquitetural do PyFolds.

Este script segue o padrão do projeto:
- imports absolutos via pacote `pyfolds`
- logging apenas em arquivo (terminal silencioso)
- fallback para dataset sintético quando MNIST não estiver disponível
- salvamento em checkpoint versionado e container `.fold`

Execução sugerida:
    PYTHONPATH=src python test/mnist_training.py train --epochs 10
    PYTHONPATH=src python test/mnist_training.py infer --fold-path outputs/mnist_training.fold
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from pyfolds import (
    FoldReader,
    FoldWriter,
    LearningMode,
    MPJRDConfig,
    MPJRDLayer,
    VersionedCheckpoint,
    __version__,
)
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
    run_name: str = "mnist_training"


@dataclass
class InferenceConfig:
    batch_size: int = 32
    test_limit: int = 128
    samples_to_show: int = 10


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


def _load_mnist(batch_size: int, train_limit: int, test_limit: int, logger) -> tuple[DataLoader, DataLoader, str]:
    if datasets is None or transforms is None:
        logger.warning("torchvision indisponível: usando dataset sintético")
        train_ds = SyntheticMNIST(train_limit)
        test_ds = SyntheticMNIST(test_limit, seed=99)
        source = "synthetic"
    else:
        try:
            tfm = transforms.Compose([transforms.ToTensor()])
            train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
            test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
            train_ds = _subset(train_ds, train_limit)
            test_ds = _subset(test_ds, test_limit)
            source = "mnist"
        except Exception as exc:  # pragma: no cover
            logger.warning("Falha ao carregar MNIST (%s): usando sintético", exc)
            train_ds = SyntheticMNIST(train_limit)
            test_ds = SyntheticMNIST(test_limit, seed=99)
            source = "synthetic"

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
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


def _save_model_as_fold(
    model: MnistMPJRDModel,
    mp_cfg: MPJRDConfig,
    train_cfg: TrainingConfig,
    dataset_source: str,
    accuracy: float,
    fold_path: Path,
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "model_cfg": mp_cfg.to_dict(),
        "n_neurons": train_cfg.n_neurons,
        "training_cfg": asdict(train_cfg),
        "dataset_source": dataset_source,
        "accuracy": accuracy,
        "version": __version__,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)

    metadata = {
        "model_type": "MnistMPJRDModel",
        "run_name": train_cfg.run_name,
        "dataset_source": dataset_source,
        "accuracy": accuracy,
        "version": __version__,
    }

    fold_path.parent.mkdir(parents=True, exist_ok=True)
    with FoldWriter(str(fold_path), compress="zstd") as writer:
        writer.add_chunk("torch_state", "TSAV", buffer.getvalue())
        writer.add_chunk("training_summary", "JSON", json.dumps(metadata).encode("utf-8"))
        writer.finalize(metadata=metadata)


def _load_model_from_fold(fold_path: Path, device: torch.device) -> tuple[MnistMPJRDModel, dict]:
    with FoldReader(str(fold_path), use_mmap=True) as reader:
        payload = reader.read_torch("torch_state", map_location=str(device))
        summary = reader.read_json("training_summary") if "training_summary" in reader.list_chunks() else {}

    mp_cfg = MPJRDConfig.from_dict(payload["model_cfg"])
    model = MnistMPJRDModel(mp_cfg, n_neurons=int(payload["n_neurons"]), device=device).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, summary


def run_training(cfg: TrainingConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrainingConfig()
    logger, log_path = setup_run_logging(app=cfg.run_name, console=False, level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mp_cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        plastic=True,
        u0=0.2,
        R0=1.0,
        U=0.3,
        theta_init=1.5,
        theta_min=0.5,
        theta_max=6.0,
        homeostasis_eta=0.1,
        dead_neuron_penalty=1.0,
        w_scale=3.5,
        neuromod_mode="surprise",
    )
    model = MnistMPJRDModel(mp_cfg, n_neurons=cfg.n_neurons, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader, source = _load_mnist(cfg.batch_size, cfg.train_limit, cfg.test_limit, logger)
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
    checkpoint_path = output_dir / f"{cfg.run_name}_checkpoint.pt"
    fold_path = output_dir / f"{cfg.run_name}.fold"

    VersionedCheckpoint(model=model, version=__version__).save(
        str(checkpoint_path), extra_metadata={"source": source, "accuracy": acc}
    )
    _save_model_as_fold(model, mp_cfg, cfg, source, acc, fold_path)

    logger.info("checkpoint=%s accuracy=%.2f", checkpoint_path, acc)
    logger.info("fold_path=%s", fold_path)

    return {
        "accuracy": acc,
        "dataset": source,
        "log_path": str(log_path),
        "checkpoint": str(checkpoint_path),
        "fold_path": str(fold_path),
    }


def run_inference(fold_path: str, cfg: InferenceConfig | None = None) -> dict[str, object]:
    cfg = cfg or InferenceConfig()
    logger, log_path = setup_run_logging(app="mnist_inference", console=False, level="INFO")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, summary = _load_model_from_fold(Path(fold_path), device=device)
    _, test_loader, source = _load_mnist(cfg.batch_size, train_limit=cfg.test_limit, test_limit=cfg.test_limit, logger=logger)

    model.hidden.set_mode(LearningMode.INFERENCE)
    correct = 0
    total = 0
    examples: list[dict[str, int]] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, mode=LearningMode.INFERENCE)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.shape[0])

            if len(examples) < cfg.samples_to_show:
                for real, predicted in zip(labels.tolist(), pred.tolist()):
                    examples.append({"label": int(real), "pred": int(predicted)})
                    if len(examples) >= cfg.samples_to_show:
                        break

    accuracy = 0.0 if total == 0 else (100.0 * correct / total)
    logger.info("inference_source=%s accuracy=%.2f total=%d", source, accuracy, total)

    return {
        "accuracy": accuracy,
        "total": total,
        "examples": examples,
        "source": source,
        "fold_summary": summary,
        "log_path": str(log_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino e inferência MNIST com MPJRD")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Executa treinamento")
    train_parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs, help="Número de épocas")
    train_parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size, help="Tamanho do batch")
    train_parser.add_argument(
        "--learning-rate", type=float, default=TrainingConfig.learning_rate, help="Taxa de aprendizado"
    )
    train_parser.add_argument("--n-neurons", type=int, default=TrainingConfig.n_neurons, help="Neurônios MPJRD")
    train_parser.add_argument("--train-limit", type=int, default=TrainingConfig.train_limit, help="Limite treino")
    train_parser.add_argument("--test-limit", type=int, default=TrainingConfig.test_limit, help="Limite teste")
    train_parser.add_argument("--run-name", default=TrainingConfig.run_name, help="Nome base dos artefatos")

    infer_parser = subparsers.add_parser("infer", help="Executa inferência a partir de .fold")
    infer_parser.add_argument("--fold-path", required=True, help="Caminho do arquivo .fold")
    infer_parser.add_argument("--batch-size", type=int, default=InferenceConfig.batch_size, help="Tamanho do batch")
    infer_parser.add_argument("--test-limit", type=int, default=InferenceConfig.test_limit, help="Limite de teste")
    infer_parser.add_argument(
        "--samples-to-show", type=int, default=InferenceConfig.samples_to_show, help="Qtd de exemplos para exibir"
    )

    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    command = args.command or "train"

    if command == "infer":
        result = run_inference(
            fold_path=args.fold_path,
            cfg=InferenceConfig(
                batch_size=args.batch_size,
                test_limit=args.test_limit,
                samples_to_show=args.samples_to_show,
            ),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        result = run_training(
            TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                n_neurons=args.n_neurons,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
                run_name=args.run_name,
            )
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
