from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch
from torch import Tensor


class CognitiveInput(Protocol):
    """Protocol for Noetic-to-PyFolds cognitive input tensors.

    Required tensor shapes:
      - concept_embeddings: (batch, seq_len, concept_dim)
      - confidence: (batch, seq_len)
      - surprise: (batch, seq_len)
    """

    @property
    def concept_embeddings(self) -> Tensor: ...

    @property
    def confidence(self) -> Tensor: ...

    @property
    def surprise(self) -> Tensor: ...


@dataclass
class PyFoldsOutput:
    spikes: Tensor
    membrane_potential: Tensor
    dendritic_states: Tensor
    cognitive_feedback: Optional[Tensor] = None

    def to_noetic(self) -> Dict[str, Tensor]:
        payload: Dict[str, Tensor] = {
            "spikes": self.spikes,
            "membrane_potential": self.membrane_potential,
            "dendritic_states": self.dendritic_states,
        }
        if self.cognitive_feedback is not None:
            payload["cognitive_feedback"] = self.cognitive_feedback
        return payload


@dataclass
class CognitiveBatch:
    """Concrete helper implementing :class:`CognitiveInput`."""

    concept_embeddings: Tensor
    confidence: Tensor
    surprise: Tensor

    def validate(self) -> None:
        if self.concept_embeddings.ndim != 3:
            raise ValueError("concept_embeddings must have shape (batch, seq_len, concept_dim)")
        if self.confidence.shape != self.concept_embeddings.shape[:2]:
            raise ValueError("confidence must have shape (batch, seq_len)")
        if self.surprise.shape != self.concept_embeddings.shape[:2]:
            raise ValueError("surprise must have shape (batch, seq_len)")

