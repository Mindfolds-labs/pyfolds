from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from models.model_py import ModelPy, ModelPyConfig
from models.model_wave import ModelWave, ModelWaveConfig

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover - ambiente sem torchvision
    torchvision = None
    transforms = None


@dataclass
class ExperimentConfig:
    dataset: str
    model: str
    batch_size: int
    lr: float
    hidden_dim: int
    timesteps: int
    optimizer: str = "adam"
    weight_decay: float = 0.0
    epochs: int = 1


@dataclass
class ExperimentResult:
    config: dict[str, Any]
    final_loss: float
    test_acc_pct: float
    train_acc_pct: float
    spike_rate: float
    elapsed_s: float
    steps_per_epoch: int
    avg_batch_size: float
    last_batch_size: int
    throughput_samples_s: float
    cost_score: float


def _make_synthetic_dataset(dataset: str, train: bool) -> TensorDataset:
    if dataset == "mnist":
        n = 1024 if train else 256
        x = torch.rand(n, 1, 28, 28)
    elif dataset == "cifar10":
        n = 1024 if train else 256
        x = torch.rand(n, 3, 32, 32)
    else:
        raise ValueError("dataset deve ser 'mnist' ou 'cifar10'")

    y = torch.randint(0, 10, (x.size(0),))
    return TensorDataset(x, y)


def _build_loaders(
    dataset: str,
    batch_size: int,
    *,
    data_root: str,
    simulate: bool,
    train_subset: int,
    test_subset: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    dataset = dataset.lower()

    if not simulate and torchvision is not None and transforms is not None:
        if dataset == "mnist":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            train_ds = torchvision.datasets.MNIST(
                root=data_root, train=True, download=True, transform=transform
            )
            test_ds = torchvision.datasets.MNIST(
                root=data_root, train=False, download=True, transform=transform
            )
            input_dim = 28 * 28
        elif dataset == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ]
            )
            train_ds = torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=transform
            )
            test_ds = torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform
            )
            input_dim = 32 * 32 * 3
        else:
            raise ValueError("dataset deve ser 'mnist' ou 'cifar10'")
    else:
        train_ds = _make_synthetic_dataset(dataset, train=True)
        test_ds = _make_synthetic_dataset(dataset, train=False)
        input_dim = 28 * 28 if dataset == "mnist" else 32 * 32 * 3

    train_ds = Subset(train_ds, range(min(len(train_ds), train_subset)))
    test_ds = Subset(test_ds, range(min(len(test_ds), test_subset)))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, input_dim


def _build_model(cfg: ExperimentConfig, input_dim: int) -> nn.Module:
    if cfg.model == "py":
        return ModelPy(
            ModelPyConfig(input_dim=input_dim, hidden_dim=cfg.hidden_dim, num_classes=10)
        )
    if cfg.model == "wave":
        return ModelWave(
            ModelWaveConfig(
                input_dim=input_dim,
                hidden_dim=cfg.hidden_dim,
                num_classes=10,
                timesteps=cfg.timesteps,
            )
        )
    raise ValueError("model deve ser 'py' ou 'wave'")


def _build_optimizer(cfg: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            total += int(y.size(0))
            correct += int((pred == y).sum().item())
    return 100.0 * correct / max(1, total)


def run_experiment(
    cfg: ExperimentConfig,
    *,
    data_root: str,
    simulate: bool,
    train_subset: int,
    test_subset: int,
    num_workers: int,
    device: str,
) -> ExperimentResult:
    dev = torch.device(device)
    train_loader, test_loader, input_dim = _build_loaders(
        cfg.dataset,
        cfg.batch_size,
        data_root=data_root,
        simulate=simulate,
        train_subset=train_subset,
        test_subset=test_subset,
        num_workers=num_workers,
    )

    model = _build_model(cfg, input_dim).to(dev)
    optimizer = _build_optimizer(cfg, model)
    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    final_loss = 0.0
    train_acc = 0.0
    avg_spike_rate = 0.0
    steps_total = 0
    samples_total = 0
    batch_sizes: list[int] = []

    for _ in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        spike_rates = []
        state = None

        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            logits, out = model(x, state=state)
            state = model.detach_state(out.get("state"))

            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
            spike_rates.append(float(out.get("spike_rate", 0.0)))

            steps_total += 1
            samples_total += int(y.size(0))
            batch_sizes.append(int(y.size(0)))

        final_loss = epoch_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        avg_spike_rate = sum(spike_rates) / max(1, len(spike_rates))

    elapsed_s = time.perf_counter() - t0
    test_acc = _evaluate(model, test_loader, dev)
    throughput = samples_total / max(1e-9, elapsed_s)

    # score simples: prioriza acurácia, penaliza custo de execução.
    cost_score = test_acc - (elapsed_s * 0.2)

    return ExperimentResult(
        config=asdict(cfg),
        final_loss=final_loss,
        test_acc_pct=test_acc,
        train_acc_pct=train_acc,
        spike_rate=avg_spike_rate,
        elapsed_s=elapsed_s,
        steps_per_epoch=max(1, steps_total // max(1, cfg.epochs)),
        avg_batch_size=(sum(batch_sizes) / max(1, len(batch_sizes))),
        last_batch_size=batch_sizes[-1] if batch_sizes else 0,
        throughput_samples_s=throughput,
        cost_score=cost_score,
    )


def build_search_space(dataset: str, epochs: int) -> list[ExperimentConfig]:
    models = ["py", "wave"]
    batch_sizes = [32, 64, 128]
    lrs = [1e-3, 5e-4]
    hidden_dims = [128, 256]
    timesteps = [4, 8]
    optimizers = ["adam", "sgd"]

    configs: list[ExperimentConfig] = []
    for model, batch, lr, hidden, ts, opt in product(
        models, batch_sizes, lrs, hidden_dims, timesteps, optimizers
    ):
        if model == "py" and ts != 4:
            continue
        configs.append(
            ExperimentConfig(
                dataset=dataset,
                model=model,
                batch_size=batch,
                lr=lr,
                hidden_dim=hidden,
                timesteps=ts,
                optimizer=opt,
                epochs=epochs,
            )
        )
    return configs


def run_tuning(
    datasets: list[str],
    *,
    epochs: int,
    data_root: str,
    simulate: bool,
    train_subset: int,
    test_subset: int,
    num_workers: int,
    device: str,
    top_k: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "meta": {
            "epochs": epochs,
            "simulate": simulate,
            "device": device,
            "train_subset": train_subset,
            "test_subset": test_subset,
        },
        "datasets": {},
    }

    for dataset in datasets:
        results = []
        for cfg in build_search_space(dataset, epochs=epochs):
            res = run_experiment(
                cfg,
                data_root=data_root,
                simulate=simulate,
                train_subset=train_subset,
                test_subset=test_subset,
                num_workers=num_workers,
                device=device,
            )
            results.append(asdict(res))

        ranked = sorted(results, key=lambda item: item["cost_score"], reverse=True)
        report["datasets"][dataset] = {
            "best": ranked[0],
            "top_k": ranked[:top_k],
            "total_trials": len(results),
            "batch_feedback": {
                "best_avg_batch_size": ranked[0]["avg_batch_size"],
                "best_last_batch_size": ranked[0]["last_batch_size"],
                "best_steps_per_epoch": ranked[0]["steps_per_epoch"],
                "throughput_samples_s": ranked[0]["throughput_samples_s"],
            },
        }

    return report


def _write_markdown_summary(report: dict[str, Any], md_path: Path) -> None:
    lines = [
        "# Relatório de tuning MNIST/CIFAR10",
        "",
        f"- simulate: `{report['meta']['simulate']}`",
        f"- device: `{report['meta']['device']}`",
        f"- epochs: `{report['meta']['epochs']}`",
        "",
    ]

    for dataset, info in report["datasets"].items():
        best = info["best"]
        lines.extend(
            [
                f"## Dataset: {dataset}",
                f"- Melhor acurácia de teste: **{best['test_acc_pct']:.2f}%**",
                f"- Custo (tempo): **{best['elapsed_s']:.2f}s**",
                f"- Score final (acurácia - penalidade de tempo): **{best['cost_score']:.2f}**",
                f"- Configuração vencedora: `{json.dumps(best['config'])}`",
                f"- Batch médio: `{best['avg_batch_size']:.2f}` | último batch: `{best['last_batch_size']}`",
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de treino/tuning para MNIST e CIFAR10")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--data-root", default="./.data")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--train-subset", type=int, default=1024)
    parser.add_argument("--test-subset", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", default="runs/tuning_report.json")
    parser.add_argument("--output-md", default="runs/tuning_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_tuning(
        datasets=args.datasets,
        epochs=args.epochs,
        data_root=args.data_root,
        simulate=args.simulate,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
        num_workers=args.num_workers,
        device=args.device,
        top_k=args.top_k,
    )

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown_summary(report, md_path)

    print(f"Relatório JSON: {json_path}")
    print(f"Resumo Markdown: {md_path}")


if __name__ == "__main__":
    main()
