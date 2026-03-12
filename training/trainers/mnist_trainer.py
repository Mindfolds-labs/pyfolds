from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from pyfolds.utils.types import LearningMode

from training.config.mnist import RunConfig, serialize_config, validate_run_config
from training.io.artifacts import save_backend_artifact
from training.io.layout import print_layout, setup_logger
from training.metrics.records import EpochMetrics
from training.models.factory import build_model, extract_logits
from training.utils.data import build_mnist_loaders


def _configure_external_loggers() -> None:
    logging.getLogger("pyfolds").setLevel(logging.WARNING)
    logging.getLogger("pyfolds.advanced").setLevel(logging.WARNING)
    logging.getLogger("pyfolds.core").setLevel(logging.WARNING)
    logging.getLogger("pyfolds.advanced.dendrite").setLevel(logging.ERROR)
    logging.getLogger("pyfolds.advanced.homeostasis").setLevel(logging.ERROR)
    logging.getLogger("pyfolds.advanced.neuron").setLevel(logging.ERROR)
    logging.getLogger("pyfolds.core.inhibition").setLevel(logging.ERROR)


def train_one_epoch(model, loader, criterion, optim, device, model_name: str):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    spike_rates: list[float] = []

    if hasattr(model, "set_mode"):
        model.set_mode(LearningMode.ONLINE)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, mode=LearningMode.ONLINE) if model_name == "mpjrd" else model(x)
        logits = extract_logits(out, x.size(0), device)
        loss = criterion(logits, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        if isinstance(out, dict):
            spike_rate = out.get("spike_rate", 0.0)
            if spike_rate:
                spike_rates.append(spike_rate)

    return total_loss / len(loader), 100.0 * correct / total, (sum(spike_rates) / len(spike_rates) if spike_rates else 0.0)


def evaluate(model, loader, device, model_name: str):
    model.eval()
    if hasattr(model, "set_mode"):
        model.set_mode(LearningMode.INFERENCE)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, mode=LearningMode.INFERENCE) if model_name == "mpjrd" else model(x)
            logits = extract_logits(out, x.size(0), device)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def run_mnist_training(config: RunConfig) -> int:
    validate_run_config(config)

    run_dir = Path("runs") / config.base.run_id
    logger = setup_logger(run_dir, config.base.log_file, config.base.console)
    device = torch.device(config.base.device)

    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"

    test_acc = 0.0
    epoch_loss = 0.0
    epochs_completed = 0
    best_acc = 0.0

    try:
        _configure_external_loggers()

        model, metadata, mpjrd_cfg = build_model(config, device)
        print_layout(config, metadata, mpjrd_cfg)
        logger.info("Modelo instanciado: family=%s config=%s", metadata.family, metadata.config)

        with torch.no_grad():
            _ = model(torch.zeros(1, 1, 28, 28, device=device))

        optim = torch.optim.Adam(model.parameters(), lr=config.base.lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        if config.base.resume and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optim.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_acc = float(ckpt.get("best_acc", 0.0))
            logger.info("🔄 RESUME epoch=%s", start_epoch)

        train_loader, test_loader = build_mnist_loaders(config.base.batch)
        epochs_completed = start_epoch

        with metrics_path.open("a", encoding="utf-8") as mf:
            for epoch in range(start_epoch, config.base.epochs):
                avg_loss, train_acc, avg_spike = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optim,
                    device,
                    metadata.family,
                )
                test_acc = evaluate(model, test_loader, device, metadata.family)
                best_acc = max(best_acc, test_acc)
                epoch_loss = avg_loss

                msg = (
                    f"Epoch {epoch+1}/{config.base.epochs} | loss={avg_loss:.4f} | "
                    f"train={train_acc:.2f}% | test={test_acc:.2f}% | spike={avg_spike:.6f}"
                )
                if config.base.console:
                    print(msg)
                logger.info(msg)

                if config.base.save_metrics:
                    mf.write(json.dumps(EpochMetrics(epoch + 1, avg_loss, train_acc, test_acc, avg_spike).to_dict()) + "\n")
                    mf.flush()

                if metadata.family == "mpjrd" and hasattr(model, "sleep") and not config.mpjrd.disable_homeostase:
                    logger.info("💤 Ciclo de sono (consolidação I → N)")
                    model.sleep(duration=60.0)

                if config.base.save_pt:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optim.state_dict(),
                            "best_acc": best_acc,
                        },
                        checkpoint_path,
                    )

                epochs_completed = epoch + 1

        logger.info("FINAL SUMMARY | final_acc=%.2f%% | best_acc=%.2f%% | final_loss=%.4f", test_acc, best_acc, epoch_loss)

        payload = {
            "model_state": model.state_dict(),
            "model_config": model.get_config() if hasattr(model, "get_config") else {},
            "epoch": max(0, config.base.epochs - 1),
            "best_acc": best_acc,
            "final_acc": test_acc,
            "final_loss": epoch_loss,
            "model_type": metadata.family,
            "run_id": config.base.run_id,
            "timestamp": datetime.now().isoformat(),
            "hyperparameters": {
                "batch_size": config.base.batch,
                "learning_rate": config.base.lr,
                "epochs": config.base.epochs,
                "device": config.base.device,
                "model": metadata.family,
                **metadata.config,
            },
        }

        if config.base.save_fold:
            save_backend_artifact("folds", run_dir / "model.fold", payload)
        if config.base.save_mind:
            save_backend_artifact("mind", run_dir / "model.mind", payload)

        if config.base.save_summary:
            summary = {
                "run_id": config.base.run_id,
                "backend": config.base.backend,
                "model": metadata.family,
                "epochs_requested": config.base.epochs,
                "epochs_completed": epochs_completed,
                "resume_used": config.base.resume,
                "best_acc_pct": best_acc,
                "final_acc_pct": test_acc,
                "final_loss": epoch_loss,
                "model_config": payload["model_config"],
                "train_config": serialize_config(config),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return 0

    except Exception as exc:
        logger.exception("❌ FALHA NO TREINO")
        if config.base.console:
            print(f"❌ FALHA NO TREINO: {exc}")
        if config.base.save_summary:
            fail_summary = {
                "run_id": config.base.run_id,
                "model": config.base.model,
                "status": "failed",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
                "train_config": asdict(config.base),
            }
            summary_path.write_text(json.dumps(fail_summary, indent=2), encoding="utf-8")
        return 1
