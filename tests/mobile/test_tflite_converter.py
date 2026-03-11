import pytest

from pyfolds.mobile.tflite_converter import (
    NumericDiscrepancy,
    _classify_model,
    diagnose_tflite_model,
    validate_tflite_model,
)


def test_validate_tflite_missing_file():
    with pytest.raises(FileNotFoundError, match="TFLite model file not found"):
        validate_tflite_model("/tmp/not_found.tflite")


def test_diagnose_tflite_missing_file():
    with pytest.raises(FileNotFoundError, match="TFLite model file not found"):
        diagnose_tflite_model("/tmp/not_found.tflite")


def test_classification_confiavel_without_risk_flags():
    result = _classify_model(
        ops={"contains_custom": False, "has_select_tf_ops": False},
        io_details=[{"dtype": "numpy.float32"}],
        discrepancy=NumericDiscrepancy(measured=False, max_abs_error=None, mean_abs_error=None, threshold=0.05),
    )
    assert result == "confiável"


def test_classification_degradado_with_aggressive_quantization():
    result = _classify_model(
        ops={"contains_custom": False, "has_select_tf_ops": False},
        io_details=[{"dtype": "numpy.int8"}],
        discrepancy=NumericDiscrepancy(measured=False, max_abs_error=None, mean_abs_error=None, threshold=0.05),
    )
    assert result == "degradado semanticamente"
