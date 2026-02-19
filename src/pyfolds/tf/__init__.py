"""TensorFlow backend for PyFolds.

This module is optional and only available when TensorFlow is installed.
"""

from __future__ import annotations

from importlib.util import find_spec

_TF_ERROR = (
    "TensorFlow backend requested, but TensorFlow is not installed. "
    "Install it with `pip install tensorflow` (or `tensorflow-cpu`) and retry."
)

_TF_AVAILABLE = find_spec("tensorflow") is not None

if _TF_AVAILABLE:
    from .layers import MPJRDTFLayer
    from .neuron import MPJRDTFNeuronCell

    __all__ = ["MPJRDTFNeuronCell", "MPJRDTFLayer"]
else:
    __all__ = []


def __getattr__(name: str):
    if name in {"MPJRDTFNeuronCell", "MPJRDTFLayer"}:
        raise ImportError(_TF_ERROR)
    raise AttributeError(f"module 'pyfolds.tf' has no attribute {name!r}")
