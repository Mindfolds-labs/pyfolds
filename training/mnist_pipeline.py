from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.model_py import ModelPy
from models.model_wave import ModelWave
from serialization.folds_io import load_model_fold, save_model_fold
from serialization.mind_io import load_model_mind, save_model_mind

try:
    import torchvision
    import torchvision.transforms as transforms
except Exception:
    torchvision = None
    transforms = None


@dataclass
class TrainArgs:
    backend: str
    model: str
    epochs: int
    batch: int
    lr: float
    run_id: str
    resume: bool
    device: str
    console: bool
    log_level: str
    log_file: str
    sheer_cmd: str = ""


def _setup_logger(run_dir: Path, *, log_level: str, log_file: str, console: bool = False) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"mnist.{run_dir.name}")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = run_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    f_handler = logging.FileHandler(log_path, encoding="utf-8")
    f_handler.setFormatter(fmt)
    logger.addHandler(f_handler)

    if console:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)

    return logger


def _build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    if torchvision is not None and transforms is not None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        try:
            train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            test_ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(test_ds, batch_size=1000)
        except Exception:
            pass

    x_train = torch.rand(2048, 1, 28, 28)
    y_train = torch.randint(0, 10, (2048,))
    x_test = torch.rand(512, 1, 28, 28)
    y_test = torch.randint(0, 10, (512,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=512),
    )


def _save_backend(backend: str, model_path: Path, payload: dict[str, Any]) -> None:
    if backend == "folds":
        save_model_fold(model_path, payload)
    else:
        save_model_mind(model_path, payload)


def _load_backend(backend: str, model_path: Path, device: str) -> dict[str, Any]:
    if backend == "folds":
        return load_model_fold(model_path, map_location=device)
    return load_model_mind(model_path, map_location=device)


def _write_adr_err(run_dir: Path, exc: Exception) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = run_dir / f"ADR-ERR-{ts}.md"
    p.write_text(
        "\n".join(
            [
                "# ADR-ERR",
                f"Timestamp: {ts}",
                f"Exception: {type(exc).__name__}: {exc}",
                "",
                "## Stacktrace",
                "```",
                traceback.format_exc(),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return p


def _write_issue(run_dir: Path, exc: Exception, adr_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    issue_dir = Path("docs/issues")
    issue_dir.mkdir(parents=True, exist_ok=True)
    p = issue_dir / f"BUG-{ts}.md"
    p.write_text(
        "\n".join(
            [
                "# Bug report (auto-gerado)",
                f"- run_id: {run_dir.name}",
                f"- erro: {type(exc).__name__}: {exc}",
                f"- adr: {adr_path}",
                f"- os: {platform.platform()}",
                f"- python: {sys.version}",
                "",
                "## Passos para reproduzir",
                f"1. Executar script no run-id {run_dir.name}",
                "2. Verificar train.log e ADR-ERR",
            ]
        ),
        encoding="utf-8",
    )
    return p


def _run_sheer_audit(cmd: str, logger: logging.Logger) -> None:
    if not cmd.strip():
        return
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.stdout:
        logger.info("[sheer] stdout:\n%s", proc.stdout.rstrip())
    if proc.stderr:
        logger.warning("[sheer] stderr:\n%s", proc.stderr.rstrip())
    if proc.returncode != 0:
        logger.error("[sheer] returncode=%s", proc.returncode)


def run_training(args: TrainArgs) -> int:
    run_dir = Path("runs") / args.run_id
    logger = _setup_logger(run_dir, log_level=args.log_level, log_file=args.log_file, console=args.console)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    artifact_path = run_dir / ("model.fold" if args.backend == "folds" else "model.mind")

    test_acc = 0.0
    epoch_loss = 0.0
    epochs_completed = 0
    best_acc = 0.0

    try:
        device = torch.device(args.device)
        model = ModelPy() if args.model == "py" else ModelWave()
        model = model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        if args.resume and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optim.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_acc = float(ckpt.get("best_acc", 0.0))
            logger.info("RESUME epoch=%s", start_epoch)
            if artifact_path.exists():
                _load_backend(args.backend, artifact_path, args.device)

        train_loader, test_loader = _build_loaders(args.batch)
        epochs_completed = start_epoch

        with metrics_path.open("a", encoding="utf-8") as mf:
            for epoch in range(start_epoch, args.epochs):
                model.train()
                total_loss, total, correct = 0.0, 0, 0
                state = None
                spike_rates: list[float] = []

                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    logits, out = model(x, state=state)
                    state = model.detach_state(out.get("state"))
                    loss = criterion(logits, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total_loss += float(loss.item())
                    pred = logits.argmax(dim=1)
                    total += y.size(0)
                    correct += int((pred == y).sum().item())
                    spike_rates.append(float(out.get("spike_rate", 0.0)))

                train_acc = 100.0 * correct / max(total, 1)
                model.eval()
                t_total, t_correct = 0, 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        logits, _ = model(x)
                        pred = logits.argmax(dim=1)
                        t_total += y.size(0)
                        t_correct += int((pred == y).sum().item())
                test_acc = 100.0 * t_correct / max(t_total, 1)
                best_acc = max(best_acc, test_acc)
                epoch_loss = total_loss / max(len(train_loader), 1)
                spike_rate = sum(spike_rates) / max(len(spike_rates), 1)

                logger.info(
                    "Epoch %s/%s | loss=%.4f | train=%.2f%% | test=%.2f%%",
                    epoch + 1,
                    args.epochs,
                    epoch_loss,
                    train_acc,
                    test_acc,
                )
                logger.info("Neuron metrics | spike_rate=%.6f", spike_rate)

                mf.write(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "loss": epoch_loss,
                            "train_acc_pct": train_acc,
                            "test_acc_pct": test_acc,
                            "spike_rate": spike_rate,
                        }
                    )
                    + "\n"
                )
                mf.flush()

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optim.state_dict(),
                        "best_acc": best_acc,
                    },
                    checkpoint_path,
                )
                _save_backend(args.backend, artifact_path, {"model_state": model.state_dict(), "epoch": epoch})
                epochs_completed = epoch + 1

        _run_sheer_audit(args.sheer_cmd, logger)

        summary = {
            "run_id": args.run_id,
            "backend": args.backend,
            "model": args.model,
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "resume_used": args.resume,
            "best_acc_pct": best_acc,
            "final_acc_pct": test_acc,
            "final_loss": epoch_loss,
            "model_config": model.get_config(),
            "train_config": asdict(args),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("FINAL SUMMARY | final_acc=%.2f%% | best_acc=%.2f%% | final_loss=%.4f", test_acc, best_acc, epoch_loss)
        return 0
    except Exception as exc:
        logger.exception("FALHA NO TREINO (stacktrace completo abaixo).")
        adr = _write_adr_err(run_dir, exc)
        _write_issue(run_dir, exc, adr)
        fail_summary = {
            "run_id": args.run_id,
            "backend": args.backend,
            "model": args.model,
            "resume_used": args.resume,
            "epochs_completed": epochs_completed,
            "final_acc_pct": test_acc,
            "final_loss": epoch_loss,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "train_config": asdict(args),
            "adr_error": adr.name,
        }
        summary_path.write_text(json.dumps(fail_summary, indent=2), encoding="utf-8")
        raise
