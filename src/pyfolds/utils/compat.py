"""Optional dependency compatibility helpers for PyFolds."""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required but unavailable."""


def _import_optional(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


@lru_cache(maxsize=None)
def has_tensorboard() -> bool:
    return _import_optional("torch.utils.tensorboard") is not None


@lru_cache(maxsize=None)
def has_prometheus() -> bool:
    return _import_optional("prometheus_client") is not None


@lru_cache(maxsize=None)
def has_psutil() -> bool:
    return _import_optional("psutil") is not None


@lru_cache(maxsize=None)
def has_jaxtyping() -> bool:
    return _import_optional("jaxtyping") is not None


@lru_cache(maxsize=None)
def has_beartype() -> bool:
    return _import_optional("beartype") is not None


@lru_cache(maxsize=None)
def has_zstandard() -> bool:
    return _import_optional("zstandard") is not None


@lru_cache(maxsize=None)
def has_onnx() -> bool:
    return _import_optional("onnx") is not None


@lru_cache(maxsize=None)
def has_tensorflow() -> bool:
    return _import_optional("tensorflow") is not None


def _require(module: str, feature: str, extra: str) -> ModuleType:
    mod = _import_optional(module)
    if mod is None:
        raise OptionalDependencyError(
            f"{feature} requires optional dependency '{module}'. "
            f"Install with: pip install 'pyfolds[{extra}]'"
        )
    return mod


def require_tensorboard() -> ModuleType:
    return _require("torch.utils.tensorboard", "TensorBoard integration", "telemetry")


def require_prometheus() -> ModuleType:
    return _require("prometheus_client", "Prometheus exporter", "observability")
