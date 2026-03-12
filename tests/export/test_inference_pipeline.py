from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyfolds.export.onnx_exporter import export_to_onnx
from pyfolds.export.tf_converter import convert_onnx_to_tf, validate_tf_saved_model
from pyfolds.mobile.tflite_converter import convert_saved_model_to_tflite, validate_tflite_model

_LEVEL_ORDER = {"A": 3, "B": 2, "C": 1}
_LEVEL_TOLERANCES = {
    "A": (1e-6, 1e-5),
    "B": (1e-4, 1e-3),
    "C": (1e-2, 1e-1),
}


def _achieved_level(a: np.ndarray, b: np.ndarray) -> str:
    abs_err = float(np.max(np.abs(a - b)))
    denom = np.maximum(np.abs(b), 1e-12)
    rel_err = float(np.max(np.abs(a - b) / denom))
    for level in ("A", "B", "C"):
        atol, rtol = _LEVEL_TOLERANCES[level]
        if abs_err <= atol and rel_err <= rtol:
            return level
    return "below-C"


def _assert_declared_level_not_overstated(*, component: str, declared: str, achieved: str) -> None:
    assert achieved in _LEVEL_ORDER, f"{component} ficou abaixo de C (achieved={achieved})."
    assert _LEVEL_ORDER[achieved] >= _LEVEL_ORDER[declared], (
        f"{component} declara nível {declared}, mas entrega apenas {achieved}."
    )


@pytest.mark.parametrize("target_level", ["A", "B", "C"])
def test_pipeline_pt_onnx_tf_signature_and_numeric_equivalence(tmp_path: Path, target_level: str):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("tensorflow")
    pytest.importorskip("onnx_tf")

    torch.manual_seed(123)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    ).eval()

    with torch.no_grad():
        model[0].weight.fill_(0.25)
        model[0].bias.fill_(0.1)
        model[2].weight.fill_(0.5)
        model[2].bias.fill_(0.2)

    sample = torch.tensor([[0.2, -0.1, 0.5, 0.8]], dtype=torch.float32)
    ref = model(sample).detach().cpu().numpy()

    onnx_path = export_to_onnx(
        model,
        tmp_path / "pt_model.onnx",
        sample,
        input_names=["x"],
        output_names=["y"],
    )
    saved_model_dir = convert_onnx_to_tf(onnx_path, tmp_path / "saved_model")

    validated = validate_tf_saved_model(saved_model_dir)
    assert validated["signatures"], "SavedModel precisa expor ao menos uma assinatura."

    import tensorflow as tf

    loaded = tf.saved_model.load(str(saved_model_dir))
    infer = loaded.signatures[validated["signatures"][0]]

    sig = infer.structured_input_signature[1]
    input_key = next(iter(sig))
    out_key = next(iter(infer.structured_outputs))

    tf_out = infer(**{input_key: tf.constant(sample.detach().cpu().numpy())})[out_key].numpy()

    atol, rtol = _LEVEL_TOLERANCES[target_level]
    assert np.allclose(ref, tf_out, atol=atol, rtol=rtol)

    achieved = _achieved_level(ref, tf_out)
    _assert_declared_level_not_overstated(
        component="pipeline_pt_onnx_tf",
        declared="B",
        achieved=achieved,
    )


def test_pipeline_tf_to_tflite_signature_and_numeric_check(tmp_path: Path):
    tf = pytest.importorskip("tensorflow")

    class AddOne(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([None, 4], tf.float32, name="x")])
        def __call__(self, x):
            return {"y": (x * 2.0) + 1.0}

    module = AddOne()
    saved_model_dir = tmp_path / "saved"
    tf.saved_model.save(module, str(saved_model_dir))

    check = validate_tf_saved_model(saved_model_dir)
    assert "serving_default" in check["signatures"]

    report = convert_saved_model_to_tflite(saved_model_dir, tmp_path / "model.tflite")
    tflite_path = Path(report["model_path"])
    assert tflite_path.exists()

    validation = validate_tflite_model(tflite_path)
    assert validation["valid"] is True

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    sample = np.array([[1.5, 0.0, -2.0, 3.0]], dtype=np.float32)
    expected = (sample * 2.0) + 1.0

    interpreter.resize_tensor_input(inp["index"], sample.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(inp["index"], sample)
    interpreter.invoke()
    got = interpreter.get_tensor(out["index"])

    achieved = _achieved_level(expected, got)
    _assert_declared_level_not_overstated(
        component="pipeline_tf_tflite",
        declared="A",
        achieved=achieved,
    )

    atol, rtol = _LEVEL_TOLERANCES["A"]
    assert np.allclose(expected, got, atol=atol, rtol=rtol)
