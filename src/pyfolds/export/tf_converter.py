"""ONNX -> TensorFlow conversion helpers with optional dependency guards."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class TFConversionError(RuntimeError):
    """Raised when ONNX -> TensorFlow conversion fails."""


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow não está instalado. Use `pip install tensorflow-cpu`."
        ) from exc
    return tf


def _require_onnx_tf_backend():
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as exc:
        raise ImportError(
            "Conversão ONNX->TF requer dependências opcionais `onnx` e `onnx-tf`."
        ) from exc
    return onnx, prepare


def convert_onnx_to_tf(
    onnx_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    strict: bool = False,
) -> Path:
    """Converte um arquivo ONNX para TensorFlow SavedModel."""
    src = Path(onnx_path).expanduser().resolve()
    dst = Path(output_dir).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Arquivo ONNX inválido: {src}")

    onnx, prepare = _require_onnx_tf_backend()

    dst.mkdir(parents=True, exist_ok=True)

    try:
        model = onnx.load(str(src))
        tf_rep = prepare(model, strict=strict)
        tf_rep.export_graph(str(dst))
    except Exception as exc:
        raise TFConversionError(
            "Falha na conversão ONNX->TensorFlow; possível operador incompatível. "
            f"Detalhe: {exc}"
        ) from exc

    logger.info("SavedModel gerado em %s", dst)
    return dst


def validate_tf_saved_model(
    saved_model_dir: Union[str, Path],
    sample_input: Optional[Any] = None,
) -> Dict[str, Any]:
    """Valida carregamento do SavedModel e assinatura básica."""
    tf = _require_tf()
    model_dir = Path(saved_model_dir).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"SavedModel não encontrado: {model_dir}")

    loaded = tf.saved_model.load(str(model_dir))
    signatures = list(getattr(loaded, "signatures", {}).keys())

    executed = False
    if sample_input is not None and signatures:
        try:
            fn = loaded.signatures[signatures[0]]
            _ = fn(sample_input)
            executed = True
        except Exception as exc:
            logger.warning("Falha no smoke test de inferência do SavedModel: %s", exc)

    return {"path": str(model_dir), "signatures": signatures, "sample_execution_ok": executed}
