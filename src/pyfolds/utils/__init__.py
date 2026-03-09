"""Utility helpers exported by :mod:`pyfolds.utils`."""

from .compat import (
    OptionalDependencyError,
    has_beartype,
    has_jaxtyping,
    has_onnx,
    has_prometheus,
    has_psutil,
    has_tensorboard,
    has_tensorflow,
    has_zstandard,
    require_prometheus,
    require_tensorboard,
)
from .device import DeviceManager, ensure_device, get_device, infer_device
from .logging import (
    TRACE_LEVEL,
    PyFoldsLogger,
    build_log_path,
    get_logger,
    next_log_path,
    setup_run_logging,
    trace,
)
from .math import calculate_vc_dimension, clamp_R, clamp_rate, safe_div, xavier_init
from .types import (
    AdaptationConfig,
    AdaptationOutput,
    ConnectionType,
    LearningMode,
    ModeConfig,
    normalize_learning_mode,
)
from .validation import validate_device_consistency, validate_input

__all__ = [
    "safe_div",
    "clamp_rate",
    "clamp_R",
    "xavier_init",
    "calculate_vc_dimension",
    "infer_device",
    "ensure_device",
    "get_device",
    "DeviceManager",
    "LearningMode",
    "ConnectionType",
    "ModeConfig",
    "AdaptationOutput",
    "AdaptationConfig",
    "normalize_learning_mode",
    "get_logger",
    "PyFoldsLogger",
    "trace",
    "TRACE_LEVEL",
    "setup_run_logging",
    "next_log_path",
    "build_log_path",
    "validate_input",
    "validate_device_consistency",
    "OptionalDependencyError",
    "has_tensorboard",
    "has_prometheus",
    "has_psutil",
    "has_jaxtyping",
    "has_beartype",
    "has_zstandard",
    "has_onnx",
    "has_tensorflow",
    "require_tensorboard",
    "require_prometheus",
]
