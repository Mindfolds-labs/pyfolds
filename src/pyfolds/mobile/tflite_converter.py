"""TensorFlow Lite conversion utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class NumericDiscrepancy:
    measured: bool
    max_abs_error: Optional[float]
    mean_abs_error: Optional[float]
    threshold: float


@dataclass
class TFLiteDiagnosticReport:
    valid: bool
    model_path: str
    classification: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    ops: Dict[str, Any]
    quantization: Dict[str, Any]
    numeric_discrepancy: Dict[str, Any]


def _require_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow backend requested, but TensorFlow is not installed. "
            "Install it with `pip install tensorflow` (or `tensorflow-cpu`) and retry."
        ) from exc
    return tf


def _normalize_dtype_name(dtype: Any) -> str:
    return str(dtype).replace("<class '", "").replace("'>", "")


def _extract_io_details(tensor_details: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    details = []
    for item in tensor_details:
        quantization = item.get("quantization_parameters", {})
        scales = quantization.get("scales", [])
        zero_points = quantization.get("zero_points", [])
        details.append(
            {
                "name": item.get("name"),
                "shape": list(item.get("shape", [])),
                "dtype": _normalize_dtype_name(item.get("dtype")),
                "quantized": bool(len(scales) > 0),
                "quantization": {
                    "scales": scales.tolist() if hasattr(scales, "tolist") else list(scales),
                    "zero_points": (
                        zero_points.tolist()
                        if hasattr(zero_points, "tolist")
                        else list(zero_points)
                    ),
                },
            }
        )
    return details


def _extract_ops_details(interpreter: Any) -> Dict[str, Any]:
    ops_details = interpreter._get_ops_details()  # pylint: disable=protected-access
    op_names = [item.get("op_name", "UNKNOWN") for item in ops_details]
    contains_custom = any(name in {"CUSTOM", "DELEGATE"} for name in op_names)
    has_select_tf_ops = any("Flex" in name for name in op_names)
    return {
        "total": len(op_names),
        "unique": sorted(set(op_names)),
        "contains_custom": contains_custom,
        "has_select_tf_ops": has_select_tf_ops,
    }


def _numeric_discrepancy(
    tf: Any,
    interpreter: Any,
    reference_saved_model_dir: Optional[Union[str, Path]] = None,
) -> NumericDiscrepancy:
    threshold = 0.05
    if reference_saved_model_dir is None:
        return NumericDiscrepancy(False, None, None, threshold)

    model = tf.saved_model.load(str(Path(reference_saved_model_dir).expanduser().resolve()))
    infer = model.signatures["serving_default"]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tf_inputs = {}
    for item in input_details:
        shape = [dim if dim > 0 else 1 for dim in item["shape"]]
        dtype = item["dtype"]
        tensor = tf.random.uniform(shape=shape, minval=-1.0, maxval=1.0, dtype=tf.float32)
        if dtype != tf.float32.as_numpy_dtype:
            tensor = tf.cast(tensor, tf.as_dtype(dtype))
        input_name = item["name"].split(":")[0]
        tf_inputs[input_name] = tensor
        interpreter.set_tensor(item["index"], tensor.numpy())

    tf_outputs = infer(**tf_inputs)
    interpreter.invoke()

    abs_errors: List[float] = []
    for idx, output_item in enumerate(output_details):
        tflite_output = interpreter.get_tensor(output_item["index"])
        tf_output = list(tf_outputs.values())[idx].numpy()
        abs_diff = abs(tf_output - tflite_output)
        abs_errors.extend(abs_diff.reshape(-1).tolist())

    if not abs_errors:
        return NumericDiscrepancy(True, 0.0, 0.0, threshold)
    max_abs_error = max(abs_errors)
    mean_abs_error = sum(abs_errors) / len(abs_errors)
    return NumericDiscrepancy(True, float(max_abs_error), float(mean_abs_error), threshold)


def _classify_model(
    *,
    ops: Dict[str, Any],
    io_details: List[Dict[str, Any]],
    discrepancy: NumericDiscrepancy,
) -> str:
    aggressive_quantization = any(item["dtype"] in {"numpy.int8", "numpy.uint8"} for item in io_details)
    non_float_io = any(item["dtype"] != "numpy.float32" for item in io_details)
    degraded_semantic = discrepancy.measured and (discrepancy.max_abs_error or 0.0) > discrepancy.threshold

    if degraded_semantic or aggressive_quantization:
        return "degradado semanticamente"
    if ops["contains_custom"] or ops["has_select_tf_ops"] or non_float_io:
        return "experimental"
    return "confiável"


def diagnose_tflite_model(
    tflite_path: Union[str, Path],
    *,
    reference_saved_model_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    tf = _require_tf()
    path = Path(tflite_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"TFLite model file not found: {path}")

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs = _extract_io_details(interpreter.get_input_details())
    outputs = _extract_io_details(interpreter.get_output_details())
    io_details = [*inputs, *outputs]
    ops = _extract_ops_details(interpreter)
    discrepancy = _numeric_discrepancy(tf, interpreter, reference_saved_model_dir)

    quantized_tensors = [item for item in io_details if item["quantized"]]
    quantization = {
        "applied": bool(quantized_tensors),
        "aggressive": any(item["dtype"] in {"numpy.int8", "numpy.uint8"} for item in io_details),
        "quantized_tensor_count": len(quantized_tensors),
    }
    classification = _classify_model(ops=ops, io_details=io_details, discrepancy=discrepancy)
    report = TFLiteDiagnosticReport(
        valid=True,
        model_path=str(path),
        classification=classification,
        inputs=inputs,
        outputs=outputs,
        ops=ops,
        quantization=quantization,
        numeric_discrepancy=asdict(discrepancy),
    )
    return asdict(report)


def convert_saved_model_to_tflite(
    saved_model_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    optimizations: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    src = Path(saved_model_dir).expanduser().resolve()
    dst = Path(output_path).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"SavedModel directory not found: {src}")
    tf = _require_tf()

    converter = tf.lite.TFLiteConverter.from_saved_model(str(src))
    if optimizations:
        parsed_optimizations = []
        for name in optimizations:
            if not isinstance(name, str):
                raise TypeError("Invalid optimization value: expected optimization names as strings.")
            try:
                parsed_optimizations.append(getattr(tf.lite.Optimize, name))
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported TFLite optimization `{name}`. "
                    "Use names from `tf.lite.Optimize`, e.g. `DEFAULT`."
                ) from exc
        converter.optimizations = parsed_optimizations
    tflite_model = converter.convert()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(tflite_model)
    report = diagnose_tflite_model(dst, reference_saved_model_dir=src)
    report["saved_model_dir"] = str(src)
    return report


def validate_tflite_model(
    tflite_path: Union[str, Path],
    *,
    reference_saved_model_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    path = Path(tflite_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"TFLite model file not found: {path}")
    return diagnose_tflite_model(path, reference_saved_model_dir=reference_saved_model_dir)
