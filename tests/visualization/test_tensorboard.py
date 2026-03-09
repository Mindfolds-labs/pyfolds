from pathlib import Path

import pytest

from pyfolds.visualization.tensorboard import TensorBoardLogger


def test_tensorboard_logger_smoke(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")

    logger = TensorBoardLogger(str(tmp_path / "tb"))
    logger.log_scalar("train/loss", 0.3, 1)
    logger.log_engram_metrics(step=1, wave_activity=0.7, engram_count=4, consolidation=0.9, pruning=0.1)
    logger.log_specialization_distribution({"vision": 0.8}, step=1)
    logger.flush()
    logger.close()

    assert (tmp_path / "tb").exists()
