import pytest

from pyfolds.mobile.tflite_converter import validate_tflite_model


def test_validate_tflite_missing_file():
    with pytest.raises(FileNotFoundError, match="TFLite model file not found"):
        validate_tflite_model("/tmp/not_found.tflite")
