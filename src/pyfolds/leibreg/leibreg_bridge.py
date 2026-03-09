"""Bridge from Noetic inputs into LEIBREG conceptual reasoning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pyfolds.advanced.engram import EngramBank

from .imagination import Imagination
from .leibniz_layer import LeibnizLayer
from .reg_core import REGCore
from .wave_context import extract_wave_phase
from .wordspace import WordSpace


class NoeticLeibregBridge(nn.Module):
    """Optional adapter for text/image-like modality features.

    The repo currently has no PAWP/RIVE modules, so this bridge accepts feature
    tensors directly and keeps all integrations optional.
    """

    def __init__(
        self,
        dim_text: Optional[int] = None,
        dim_image: Optional[int] = None,
        dim_concept: int = 4,
        concept_count: int = 1024,
        engram_bank: Optional[EngramBank] = None,
    ) -> None:
        super().__init__()
        self.text_layer = LeibnizLayer(dim_text, dim_concept) if dim_text is not None else None
        self.image_layer = LeibnizLayer(dim_image, dim_concept) if dim_image is not None else None
        self.wordspace = WordSpace(concept_count=concept_count, dim_base=dim_concept)
        self.reg_core = REGCore(dim=dim_concept)
        self.imagination = Imagination(engram_bank=engram_bank)

    def forward(
        self,
        texto: Optional[torch.Tensor] = None,
        imagem: Optional[torch.Tensor] = None,
        concept_ids: Optional[torch.Tensor] = None,
        wave_source: Optional[Any] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        features = []
        if texto is not None:
            if self.text_layer is None:
                raise ValueError("Bridge was created without text modality support")
            features.append(self.text_layer(texto))
        if imagem is not None:
            if self.image_layer is None:
                raise ValueError("Bridge was created without image modality support")
            features.append(self.image_layer(imagem))
        if concept_ids is not None:
            wave_phase = extract_wave_phase(wave_source) if wave_source is not None else None
            ws = self.wordspace(concept_ids, wave_phase=wave_phase)
            features.append(ws["q_total"])

        if not features:
            raise ValueError("At least one modality or concept_ids must be provided")

        x = torch.stack(features, dim=0).mean(dim=0)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        reasoned = self.reg_core(x)
        concept_point = reasoned.mean(dim=1)
        memory = self.imagination(concept_point, k=top_k)
        return {
            "concept_point": concept_point,
            "reasoned_state": reasoned,
            "memory": memory,
        }
