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
from training.utils.data import build_image_loaders, build_mnist_loaders


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




def _load_checkpoint_compat(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint inválido em {path}: esperado dict, recebido {type(ckpt).__name__}")
    return ckpt


def _extract_compatible_state(model: nn.Module, model_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    current_state = model.state_dict()
    compatible_state = {
        k: v
        for k, v in model_state.items()
        if k in current_state and tuple(v.shape) == tuple(current_state[k].shape)
    }
    return compatible_state, len(model_state) - len(compatible_state)

def run_mnist_training(config: RunConfig) -> int:
    validate_run_config(config)

    run_dir = Path("runs") / config.base.run_id
    logger = setup_logger(run_dir, config.base.log_file, config.base.console)
    if not config.base.save_log:
        # saltar FileHandler quando save_log=0 (contrato da flag)
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    device = torch.device(config.base.device)

    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"

    test_acc = 0.0
    epoch_loss = 0.0
    epochs_completed = 0
    best_acc = 0.0

    managed_logger_names = [
        "pyfolds",
        "pyfolds.advanced",
        "pyfolds.core",
        "pyfolds.advanced.dendrite",
        "pyfolds.advanced.homeostasis",
        "pyfolds.advanced.neuron",
        "pyfolds.core.inhibition",
    ]
    previous_levels = {name: logging.getLogger(name).level for name in managed_logger_names}

    try:
        _configure_external_loggers()

        model, metadata, mpjrd_cfg = build_model(config, device)
        print_layout(config, metadata, mpjrd_cfg)
        logger.info("Modelo instanciado: family=%s config=%s", metadata.family, metadata.config)

        with torch.no_grad():
            if metadata.family == "foldsnet":
                dataset = str(metadata.config.get("dataset", "mnist"))
                shape = {"mnist": (1, 28, 28), "cifar10": (3, 32, 32), "cifar100": (3, 32, 32)}[dataset]
                _ = model(torch.zeros(1, *shape, device=device))
            else:
                _ = model(torch.zeros(1, 1, 28, 28, device=device))

        optim = torch.optim.Adam(model.parameters(), lr=config.base.lr)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0

        init_checkpoint = (config.base.init_checkpoint or "").strip()
        if init_checkpoint:
            ckpt_path = Path(init_checkpoint)
            ckpt = _load_checkpoint_compat(ckpt_path, device)
            model_state = ckpt.get("model_state", ckpt)
            if not isinstance(model_state, dict):
                raise ValueError(f"init-checkpoint inválido em {ckpt_path}: campo model_state deve ser dict")
            compatible_state, incompatible_count = _extract_compatible_state(model, model_state)
            missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            logger.info(
                "🧠 WARM START carregado de %s | loaded=%d | missing=%d | unexpected=%d | incompatible=%d",
                ckpt_path,
                len(compatible_state),
                len(missing),
                len(unexpected),
                incompatible_count,
            )

        if config.base.resume and checkpoint_path.exists():
            ckpt = _load_checkpoint_compat(checkpoint_path, device)
            model_state = ckpt.get("model_state")
            optim_state = ckpt.get("optimizer_state")
            if not isinstance(model_state, dict) or optim_state is None:
                raise ValueError(
                    "Checkpoint de resume inválido: chaves obrigatórias ausentes (model_state/optimizer_state)."
                )
            compatible_state, incompatible_count = _extract_compatible_state(model, model_state)
            if incompatible_count:
                logger.warning("Resume com %d parâmetros incompatíveis por shape (serão ignorados).", incompatible_count)
            missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            optim.load_state_dict(optim_state)
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            best_acc = float(ckpt.get("best_acc", 0.0))
            logger.info(
                "🔄 RESUME epoch=%s | loaded=%d | missing=%d | unexpected=%d",
                start_epoch,
                len(compatible_state),
                len(missing),
                len(unexpected),
            )

        if metadata.family == "foldsnet":
            train_loader, test_loader = build_image_loaders(config.foldsnet.dataset, config.base.batch)
        else:
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

                spike_text = f"{avg_spike:.6f}" if metadata.family == "mpjrd" else "n/a"
                msg = (
                    f"Epoch {epoch+1}/{config.base.epochs} | loss={avg_loss:.4f} | "
                    f"train={train_acc:.2f}% | test={test_acc:.2f}% | spike={spike_text}"
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
                            "model_type": metadata.family,
                            "model_config": metadata.config,
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
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)
