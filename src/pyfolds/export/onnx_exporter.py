"""ONNX export utilities for PyFolds models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

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
    validate_export: bool = True,
    sanity_abs_limit: float = 1e6,
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

    verification = verify_onnx_model(out_path)
    if validate_export:
        validation = validate_onnx_artifact(
            model,
            out_path,
            reference_input=dummy_input,
            abs_value_limit=sanity_abs_limit,
        )
        if validation["status"] == "failed":
            raise ONNXExportError(
                "Validação pós-export ONNX falhou. "
                f"Campos divergentes: {validation['mismatch_fields']}"
            )
        if validation["status"] == "warning":
            logger.warning("Validação ONNX concluída com alertas: %s", validation)

    logger.debug("Verificação estrutural ONNX: %s", verification)
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


def _flatten_outputs(outputs: Any) -> Dict[str, Any]:
    if isinstance(outputs, dict):
        return {str(k): v for k, v in outputs.items()}
    if isinstance(outputs, (tuple, list)):
        return {f"output_{idx}": value for idx, value in enumerate(outputs)}
    return {"output_0": outputs}


def _torch_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _onnx_dtype_to_numpy(dtype: int) -> str:
    _, onnx = _require_torch_and_onnx()
    try:
        return str(np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[dtype]))
    except Exception:
        return "unknown"


def _build_synthetic_input(reference_input: Any) -> Any:
    torch, _ = _require_torch_and_onnx()
    if isinstance(reference_input, (tuple, list)):
        return type(reference_input)(_build_synthetic_input(item) for item in reference_input)
    if isinstance(reference_input, dict):
        return {key: _build_synthetic_input(value) for key, value in reference_input.items()}
    if hasattr(reference_input, "shape") and hasattr(reference_input, "dtype"):
        tensor = torch.zeros_like(reference_input)
        if tensor.numel() > 0:
            pattern = torch.linspace(-1.0, 1.0, steps=tensor.numel(), dtype=torch.float32).reshape(
                tensor.shape
            )
            if tensor.dtype.is_floating_point:
                pattern = pattern.to(dtype=tensor.dtype)
            else:
                pattern = pattern.to(dtype=torch.int64)
            tensor = tensor + pattern.to(device=tensor.device)
        return tensor
    return reference_input


def validate_onnx_artifact(
    source_model: Any,
    onnx_path: Union[str, Path],
    *,
    reference_input: Any,
    abs_value_limit: float = 1e6,
) -> Dict[str, Any]:
    """Compara metadados/outputs entre modelo origem e artefato ONNX."""
    torch, onnx = _require_torch_and_onnx()
    path = Path(onnx_path).expanduser().resolve()
    model = onnx.load(str(path))

    result: Dict[str, Any] = {
        "status": "ok",
        "mismatch_fields": [],
        "max_abs_diff": None,
        "max_rel_diff": None,
    }

    model_was_training = getattr(source_model, "training", False)
    with torch.no_grad():
        source_model.eval()
        source_outputs = _flatten_outputs(source_model(reference_input))

    for idx, output_value in enumerate(model.graph.output):
        ref = source_outputs.get(output_value.name) or source_outputs.get(f"output_{idx}")
        if ref is None:
            result["mismatch_fields"].append(f"output[{idx}].name")
            continue
        onnx_shape = [d.dim_value for d in output_value.type.tensor_type.shape.dim]
        ref_shape = list(_torch_to_numpy(ref).shape)
        if onnx_shape and any(v > 0 for v in onnx_shape) and onnx_shape != ref_shape:
            result["mismatch_fields"].append(f"output[{idx}].shape")
        onnx_dtype = _onnx_dtype_to_numpy(output_value.type.tensor_type.elem_type)
        ref_dtype = str(_torch_to_numpy(ref).dtype)
        if onnx_dtype != "unknown" and onnx_dtype != ref_dtype:
            result["mismatch_fields"].append(f"output[{idx}].dtype")

    try:
        import onnxruntime as ort

        synthetic_input = _build_synthetic_input(reference_input)
        with torch.no_grad():
            ref_outputs = _flatten_outputs(source_model(synthetic_input))

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if isinstance(synthetic_input, dict):
            ort_inputs = {k: _torch_to_numpy(v) for k, v in synthetic_input.items()}
        elif isinstance(synthetic_input, (tuple, list)):
            ort_inputs = {
                session.get_inputs()[idx].name: _torch_to_numpy(value)
                for idx, value in enumerate(synthetic_input)
            }
        else:
            ort_inputs = {session.get_inputs()[0].name: _torch_to_numpy(synthetic_input)}
        ort_outputs = {
            output.name: value
            for output, value in zip(session.get_outputs(), session.run(None, ort_inputs))
        }

        max_abs = 0.0
        max_rel = 0.0
        for idx, output in enumerate(session.get_outputs()):
            ref_value = ref_outputs.get(output.name) or ref_outputs.get(f"output_{idx}")
            if ref_value is None:
                result["mismatch_fields"].append(f"runtime.output[{idx}].missing")
                continue
            ref_np = _torch_to_numpy(ref_value).astype(np.float64)
            ort_np = np.asarray(ort_outputs[output.name], dtype=np.float64)
            if ref_np.shape != ort_np.shape:
                result["mismatch_fields"].append(f"runtime.output[{idx}].shape")
                continue
            diff = np.abs(ref_np - ort_np)
            max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
            denom = np.maximum(np.abs(ref_np), 1e-12)
            rel = diff / denom
            max_rel = max(max_rel, float(np.max(rel)) if rel.size else 0.0)

            if not np.all(np.isfinite(ort_np)):
                result["mismatch_fields"].append(f"runtime.output[{idx}].finite")
            if np.any(np.abs(ort_np) > abs_value_limit):
                result["mismatch_fields"].append(f"runtime.output[{idx}].range")

        result["max_abs_diff"] = max_abs
        result["max_rel_diff"] = max_rel
    except ImportError:
        result["status"] = "warning"
        result["mismatch_fields"].append("onnxruntime.unavailable")
    except Exception as exc:
        result["status"] = "failed"
        result["mismatch_fields"].append(f"runtime.error:{exc}")

    if model_was_training and hasattr(source_model, "train"):
        source_model.train()

    if result["status"] == "ok" and result["mismatch_fields"]:
        result["status"] = "failed"

    return result
