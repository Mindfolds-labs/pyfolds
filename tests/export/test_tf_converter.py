from pathlib import Path

import pytest

from pyfolds.export.tf_converter import convert_onnx_to_tf, validate_tf_saved_model


def test_convert_onnx_invalid_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert_onnx_to_tf(tmp_path / "none.onnx", tmp_path / "saved")


def test_validate_saved_model_invalid_path():
    with pytest.raises(FileNotFoundError):
        validate_tf_saved_model("/tmp/no_saved_model")
