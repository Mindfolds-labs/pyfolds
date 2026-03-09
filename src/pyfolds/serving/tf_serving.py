"""TensorFlow Serving helpers for SavedModel deployment."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union


def prepare_versioned_saved_model(
    saved_model_dir: Union[str, Path],
    serving_root: Union[str, Path],
    *,
    model_name: str,
    version: Union[str, int] = 1,
) -> Path:
    src = Path(saved_model_dir).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"SavedModel inválido: {src}")
    dst = Path(serving_root).expanduser().resolve() / model_name / str(version)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def build_tf_serving_command(
    *,
    serving_root: Union[str, Path],
    model_name: str,
    rest_api_port: int = 8501,
    grpc_port: int = 8500,
    docker_image: str = "tensorflow/serving",
) -> str:
    root = Path(serving_root).expanduser().resolve()
    return (
        "docker run --rm "
        f"-p {rest_api_port}:8501 -p {grpc_port}:8500 "
        f"-v \"{root}:/models/{model_name}\" "
        f"-e MODEL_NAME={model_name} {docker_image}"
    )


def build_inference_payload(instances: Sequence[Any], *, signature_name: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"instances": list(instances)}
    if signature_name:
        payload["signature_name"] = signature_name
    return payload


def payload_to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
