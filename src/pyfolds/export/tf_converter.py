"""ONNX -> TensorFlow conversion helpers with optional dependency guards."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class TFConversionError(RuntimeError):
    """Raised when ONNX -> TensorFlow conversion fails."""


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow backend requested, but TensorFlow is not installed. "
            "Install it with `pip install tensorflow` (or `tensorflow-cpu`) and retry."
        ) from exc
    return tf


def _require_onnx_tf_backend():
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as exc:
        raise ImportError(
            "ONNX->TensorFlow conversion requires optional dependencies `onnx` and `onnx-tf`."
        ) from exc
    return onnx, prepare


def _flatten_outputs(outputs: Any) -> Dict[str, Any]:
    if isinstance(outputs, dict):
        return {str(k): v for k, v in outputs.items()}
    if isinstance(outputs, (tuple, list)):
        return {f"output_{idx}": value for idx, value in enumerate(outputs)}
    return {"output_0": outputs}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _build_synthetic_input(sample_input: Any) -> Any:
    tf = _require_tf()
    if isinstance(sample_input, dict):
        return {k: _build_synthetic_input(v) for k, v in sample_input.items()}
    if isinstance(sample_input, (tuple, list)):
        return type(sample_input)(_build_synthetic_input(v) for v in sample_input)
    if isinstance(sample_input, tf.Tensor):
        total = int(np.prod(sample_input.shape)) if sample_input.shape.rank is not None else 0
        if total <= 0:
            return tf.zeros_like(sample_input)
        lin = tf.reshape(tf.linspace(-1.0, 1.0, total), sample_input.shape)
        if sample_input.dtype.is_floating:
            return tf.cast(lin, sample_input.dtype)
        return tf.cast(tf.round(lin * 10.0), sample_input.dtype)
    return sample_input


def convert_onnx_to_tf(
    onnx_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    strict: bool = False,
    validate_conversion: bool = True,
    source_onnx_outputs: Optional[Dict[str, Any]] = None,
    sample_input: Optional[Any] = None,
    sanity_abs_limit: float = 1e6,
) -> Path:
    """Converte um arquivo ONNX para TensorFlow SavedModel."""
    src = Path(onnx_path).expanduser().resolve()
    dst = Path(output_dir).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Invalid ONNX file path: {src}")

    onnx, prepare = _require_onnx_tf_backend()

    dst.mkdir(parents=True, exist_ok=True)

    try:
        model = onnx.load(str(src))
        tf_rep = prepare(model, strict=strict)
        tf_rep.export_graph(str(dst))
    except Exception as exc:
        raise TFConversionError(
            "Failed to convert ONNX to TensorFlow; possibly due to an unsupported operator. "
            f"Detail: {exc}"
        ) from exc

    if validate_conversion:
        validation = validate_tf_saved_model(
            dst,
            sample_input=sample_input,
            source_onnx_outputs=source_onnx_outputs,
            abs_value_limit=sanity_abs_limit,
        )
        if validation["status"] == "failed":
            raise TFConversionError(
                "Validation after ONNX->TF conversion failed. "
                f"Mismatch fields: {validation['mismatch_fields']}"
            )
        if validation["status"] == "warning":
            logger.warning("TF conversion validation completed with warnings: %s", validation)

    logger.info("SavedModel generated at %s", dst)
    return dst


def validate_tf_saved_model(
    saved_model_dir: Union[str, Path],
    sample_input: Optional[Any] = None,
    *,
    source_onnx_outputs: Optional[Dict[str, Any]] = None,
    abs_value_limit: float = 1e6,
) -> Dict[str, Any]:
    """Valida carregamento do SavedModel, IO metadata e sanidade de outputs."""
    tf = _require_tf()
    model_dir = Path(saved_model_dir).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"SavedModel directory not found: {model_dir}")

    loaded = tf.saved_model.load(str(model_dir))
    signatures = list(getattr(loaded, "signatures", {}).keys())

    result: Dict[str, Any] = {
        "path": str(model_dir),
        "signatures": signatures,
        "sample_execution_ok": False,
        "status": "ok",
        "mismatch_fields": [],
        "max_abs_diff": None,
        "max_rel_diff": None,
    }

    if not signatures:
        result["status"] = "warning"
        result["mismatch_fields"].append("signatures.unavailable")
        return result

    fn = loaded.signatures[signatures[0]]

    if sample_input is None:
        result["status"] = "warning"
        result["mismatch_fields"].append("sample_input.unavailable")
        return result

    synthetic_input = _build_synthetic_input(sample_input)

    try:
        outputs = fn(synthetic_input)
        result["sample_execution_ok"] = True
    except Exception as exc:
        result["status"] = "failed"
        result["mismatch_fields"].append(f"inference.error:{exc}")
        return result

    outputs_map = _flatten_outputs(outputs)

    for key, tensor in outputs_map.items():
        arr = _to_numpy(tensor)
        if not np.all(np.isfinite(arr)):
            result["mismatch_fields"].append(f"output.{key}.finite")
        if np.any(np.abs(arr) > abs_value_limit):
            result["mismatch_fields"].append(f"output.{key}.range")

    if source_onnx_outputs:
        max_abs = 0.0
        max_rel = 0.0
        for key, tf_output in outputs_map.items():
            ref = source_onnx_outputs.get(key)
            if ref is None:
                result["mismatch_fields"].append(f"io.{key}.missing_in_source")
                continue
            tf_raw = _to_numpy(tf_output)
            ref_raw = _to_numpy(ref)
            if tf_raw.shape != ref_raw.shape:
                result["mismatch_fields"].append(f"io.{key}.shape")
                continue
            if str(tf_raw.dtype) != str(ref_raw.dtype):
                result["mismatch_fields"].append(f"io.{key}.dtype")
            tf_arr = tf_raw.astype(np.float64)
            ref_arr = ref_raw.astype(np.float64)
            diff = np.abs(tf_arr - ref_arr)
            max_abs = max(max_abs, float(np.max(diff)) if diff.size else 0.0)
            denom = np.maximum(np.abs(ref_arr), 1e-12)
            rel = diff / denom
            max_rel = max(max_rel, float(np.max(rel)) if rel.size else 0.0)

        result["max_abs_diff"] = max_abs
        result["max_rel_diff"] = max_rel

    if result["status"] == "ok" and result["mismatch_fields"]:
        result["status"] = "failed"

    return result
