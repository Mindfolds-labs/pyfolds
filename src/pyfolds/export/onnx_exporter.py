"""ONNX export utilities for PyFolds models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class ONNXExportError(RuntimeError):
    """Raised when ONNX export fails."""


def _require_torch_and_onnx():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch é obrigatório para exportação ONNX. Instale `torch`.") from exc

    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "Dependência opcional ausente: `onnx`. Instale com `pip install onnx`."
        ) from exc

    return torch, onnx


def export_to_onnx(
    model: Any,
    output_path: Union[str, Path],
    dummy_input: Any,
    *,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    dynamic_axes: Optional[Mapping[str, Mapping[int, str]]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
) -> Path:
    """Exporta um modelo PyTorch/PyFolds para ONNX e valida integridade básica."""
    torch, _ = _require_torch_and_onnx()

    if not hasattr(model, "forward"):
        raise TypeError("`model` precisa expor método `forward` para exportação ONNX.")

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_was_training = getattr(model, "training", False)
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                input_names=list(input_names) if input_names else None,
                output_names=list(output_names) if output_names else None,
                dynamic_axes=dict(dynamic_axes) if dynamic_axes else None,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
            )
    except Exception as exc:
        raise ONNXExportError(
            "Falha ao exportar ONNX. Verifique operadores não suportados e shapes. "
            f"Detalhe: {exc}"
        ) from exc
    finally:
        if model_was_training and hasattr(model, "train"):
            model.train()

    verify_onnx_model(out_path)
    logger.info("Modelo exportado para ONNX: %s", out_path)
    return out_path


def verify_onnx_model(onnx_path: Union[str, Path], *, run_shape_inference: bool = True) -> Dict[str, Any]:
    """Valida modelo ONNX com checker e inferência de shape opcional."""
    path = Path(onnx_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Arquivo ONNX não encontrado: {path}")

    _, onnx = _require_torch_and_onnx()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    inferred = False
    if run_shape_inference:
        try:
            _ = onnx.shape_inference.infer_shapes(model)
            inferred = True
        except Exception as exc:
            logger.warning("Inferência de shape ONNX falhou para %s: %s", path, exc)

    return {
        "path": str(path),
        "graph_name": model.graph.name,
        "nodes": len(model.graph.node),
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "shape_inference_ok": inferred,
    }
