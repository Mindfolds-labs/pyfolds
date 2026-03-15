from __future__ import annotations

from pathlib import Path

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


def test_mnist_pipeline_foldsnet_saves_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)

    args = _base_args("foldsnet", "save_foldsnet")
    args.save_fold = 1
    args.save_mind = 1
    args.save_pt = 1

    result = run_training(args)
    assert result == 0

    run_dir = Path("runs") / args.run_id
    assert (run_dir / "model.fold").exists()
    assert (run_dir / "model.mind").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()


def test_mnist_pipeline_mpjrd_saves_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)

    args = _base_args("mpjrd", "save_mpjrd")
    args.save_fold = 1
    args.save_mind = 1
    args.save_pt = 1

    result = run_training(args)
    assert result == 0

    run_dir = Path("runs") / args.run_id
    assert (run_dir / "model.fold").exists()
    assert (run_dir / "model.mind").exists()
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()


def test_mnist_pipeline_foldsnet_resume_and_warm_start(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)

    base_args = _base_args("foldsnet", "resume_seed")
    base_args.save_pt = 1
    base_args.save_fold = 0
    base_args.save_mind = 0
    assert run_training(base_args) == 0

    run_dir = Path("runs") / base_args.run_id
    ckpt = run_dir / "checkpoint.pt"
    assert ckpt.exists()

    warm_args = _base_args("foldsnet", "warm_start_target")
    warm_args.init_checkpoint = str(ckpt)
    assert run_training(warm_args) == 0

    resume_args = _base_args("foldsnet", "resume_seed")
    resume_args.resume = True
    resume_args.epochs = 2
    resume_args.save_pt = 1
    assert run_training(resume_args) == 0


def test_resume_with_invalid_checkpoint_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("training.trainers.mnist_trainer.build_mnist_loaders", _tiny_loaders)

    run_id = "bad_resume"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "model_state": {}}, run_dir / "checkpoint.pt")

    args = _base_args("foldsnet", run_id)
    args.resume = True
    assert run_training(args) == 1
