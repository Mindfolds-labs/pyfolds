from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from training.mnist_pipeline import TrainArgs, run_training


def _tiny_loaders(batch_size: int):
    x_train = torch.rand(64, 1, 28, 28)
    y_train = torch.randint(0, 10, (64,))
    x_test = torch.rand(32, 1, 28, 28)
    y_test = torch.randint(0, 10, (32,))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False),
    )


def _base_args(model: str, run_id: str) -> TrainArgs:
    return TrainArgs(
        backend="folds",
        model=model,
        epochs=1,
        batch=16,
        lr=1e-3,
        run_id=run_id,
        resume=False,
        device="cpu",
        console=False,
        log_level="INFO",
        log_file="train.log",
        save_fold=0,
        save_mind=0,
        save_pt=0,
        save_metrics=1,
        save_summary=1,
    )


def test_mnist_pipeline_sanity_mpjrd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)
    result = run_training(_base_args("mpjrd", "test_mpjrd"))
    assert result == 0


def test_mnist_pipeline_sanity_foldsnet(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)
    result = run_training(_base_args("foldsnet", "test_foldsnet"))
    assert result == 0
