from __future__ import annotations

from training.mnist_pipeline import TrainArgs, run_training


def test_foldsnet_cifar10_pipeline_synthetic(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYFOLDS_MNIST_SYNTHETIC", "1")
    monkeypatch.setenv("PYFOLDS_SYNTH_TRAIN_SAMPLES", "16")
    monkeypatch.setenv("PYFOLDS_SYNTH_TEST_SAMPLES", "16")

    args = TrainArgs(
        backend="folds",
        model="foldsnet",
        epochs=1,
        batch=8,
        lr=1e-3,
        run_id="cifar10_smoke",
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
        foldsnet_variant="4L",
        foldsnet_dataset="cifar10",
    )

    assert run_training(args) == 0
