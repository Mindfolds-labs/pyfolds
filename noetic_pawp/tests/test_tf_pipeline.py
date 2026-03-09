from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from noetic_pawp.data.tf_pipeline import build_tf_dataset, build_nyuv2_pipeline


def test_build_tf_dataset_shapes():
    tf = pytest.importorskip("tensorflow")
    images = tf.zeros([4, 16, 16, 3])
    labels = tf.zeros([4], dtype=tf.int32)
    ds = build_tf_dataset((images, labels), batch_size=2)
    batch = next(iter(ds))
    x, y = batch
    assert x.shape[0] == 2
    assert y.shape[0] == 2


def test_nyuv2_pipeline_invalid_path():
    with pytest.raises(FileNotFoundError):
        build_nyuv2_pipeline("/tmp/does_not_exist")
