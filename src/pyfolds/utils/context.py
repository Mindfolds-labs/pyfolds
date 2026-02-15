"""Context managers utilitários para controle de execução."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from ..core.base import BaseNeuron
from .types import LearningMode


@contextmanager
def learning_mode(neuron: BaseNeuron, mode: LearningMode) -> Iterator[BaseNeuron]:
    """Altera temporariamente o modo de aprendizado de um neurônio."""
    old_mode = neuron.mode
    neuron.set_mode(mode)
    try:
        yield neuron
    finally:
        neuron.set_mode(old_mode)
