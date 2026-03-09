"""TensorFlow Lite conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow é necessário para exportar TFLite.") from exc
    return tf


def convert_saved_model_to_tflite(
    saved_model_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    optimizations: Optional[Iterable[str]] = None,
) -> Path:
    tf = _require_tf()
    src = Path(saved_model_dir).expanduser().resolve()
    dst = Path(output_path).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"SavedModel não encontrado: {src}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
    if optimizations:
        converter.optimizations = [getattr(tf.lite.Optimize, name) for name in optimizations]
    tflite_model = converter.convert()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(tflite_model)
    return dst


def validate_tflite_model(tflite_path: Union[str, Path]) -> bool:
    tf = _require_tf()
    path = Path(tflite_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Arquivo TFLite não encontrado: {path}")
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return True
