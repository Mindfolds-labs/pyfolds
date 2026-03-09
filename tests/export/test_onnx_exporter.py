from pathlib import Path

import pytest

from pyfolds.export.onnx_exporter import export_to_onnx, verify_onnx_model


def test_verify_onnx_missing_file():
    with pytest.raises(FileNotFoundError):
        verify_onnx_model("/tmp/nao_existe.onnx")


def test_export_to_onnx_smoke(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    dummy = torch.randn(1, 4)
    out = tmp_path / "model.onnx"

    generated = export_to_onnx(model, out, dummy, input_names=["x"], output_names=["y"])
    assert generated.exists()

    info = verify_onnx_model(generated)
    assert info["nodes"] >= 1
