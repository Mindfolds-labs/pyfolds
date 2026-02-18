"""Integration tests for MNIST file-only logging training script."""

from pathlib import Path

import pyfolds

from examples.mnist_file_logging import TrainConfig, run_training


def test_pyfolds_imports_are_stable():
    """Validate direct/advanced imports to detect circular import regressions."""
    assert hasattr(pyfolds, "MPJRDConfig")
    assert hasattr(pyfolds, "MPJRDLayer")
    assert hasattr(pyfolds, "LearningMode")
    assert pyfolds.MPJRDNeuronAdvanced is not None


def test_training_script_runs_end_to_end():
    """Run a tiny end-to-end training and ensure log file is produced."""
    cfg = TrainConfig(
        batch_size=16,
        epochs=1,
        n_neurons=12,
        train_limit=128,
        test_limit=64,
    )
    result = run_training(cfg)

    assert result["dataset_source"] in {"mnist", "synthetic"}
    assert result["best_acc"] >= 0.0
    log_path = Path(result["log_path"])
    assert log_path.exists()
    assert log_path.stat().st_size > 0

    checkpoint_path = Path(result["checkpoint_path"])
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 0
