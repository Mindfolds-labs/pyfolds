"""Bridge de integração entre núcleo noético e camada LEIBREG."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn

from pyfolds.leibreg.imagination import Imagination
from pyfolds.leibreg.reg_core import REGCore
from pyfolds.leibreg.wordspace import WordSpace

logger = logging.getLogger(__name__)


class NoeticLeibregBridge(nn.Module):
    """Integra texto, imagem e memória conceitual no pipeline LEIBREG."""

    def __init__(
        self,
        *,
        wordspace: WordSpace,
        reg_core: Optional[REGCore] = None,
        imagination: Optional[Imagination] = None,
        text_encoder: Optional[Callable[[Any], Tensor]] = None,
        image_encoder: Optional[Callable[[Any], Tensor]] = None,
        associative_memory: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.wordspace = wordspace
        self.reg_core = reg_core or REGCore()
        self.imagination = imagination or Imagination()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.associative_memory = associative_memory

    def _as_tensor(self, x: Any, *, label: str) -> Tensor:
        if isinstance(x, Tensor):
            return x.float()
        if x is None:
            raise ValueError(f"{label} não pode ser None neste ponto do pipeline.")
        try:
            return torch.as_tensor(x, dtype=torch.float32)
        except Exception as exc:  # pragma: no cover - proteção defensiva
            raise TypeError(f"Não foi possível converter {label} para Tensor: {type(x)!r}") from exc

    def _encode_optional(self, raw: Any, encoder: Optional[Callable[[Any], Tensor]], *, label: str) -> Optional[Tensor]:
        if raw is None:
            return None
        encoded = encoder(raw) if encoder is not None else raw
        return self._as_tensor(encoded, label=label)

    def _retrieve_memory(self, memory_query: Any) -> Optional[Tensor]:
        if memory_query is None or self.associative_memory is None:
            return None
        retriever = getattr(self.associative_memory, "retrieve", None)
        if retriever is None or not callable(retriever):
            raise AttributeError("associative_memory deve expor método callable 'retrieve'.")
        memory_vec = retriever(memory_query)
        return self._as_tensor(memory_vec, label="memory retrieval")

    def _fuse(self, *points: Optional[Tensor]) -> Tensor:
        valid = [point for point in points if point is not None]
        if not valid:
            raise ValueError("Ao menos uma modalidade (texto, imagem ou memória) deve ser fornecida.")
        stacked = torch.stack(valid, dim=0)
        return stacked.mean(dim=0)

    def forward(
        self,
        *,
        text: Any = None,
        text_features: Optional[Tensor] = None,
        image: Any = None,
        image_features: Optional[Tensor] = None,
        memory_query: Any = None,
        memory_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | None]:
        """Executa pipeline LEIBREG com suporte a entradas parciais."""
        text_base = text_features if text_features is not None else self._encode_optional(text, self.text_encoder, label="text")
        image_base = image_features if image_features is not None else self._encode_optional(image, self.image_encoder, label="image")
        memory_base = memory_features if memory_features is not None else self._retrieve_memory(memory_query)

        text_point = self.wordspace.project_text(text_base) if text_base is not None else None
        image_point = self.wordspace.project_image(image_base) if image_base is not None else None
        memory_point = self.wordspace.project_memory(memory_base) if memory_base is not None else None

        fused_point = self._fuse(text_point, image_point, memory_point)

        # REGCore espera [batch, n_tokens, 4], então adicionamos eixo de sequência unitário.
        reg_input = fused_point.unsqueeze(-2)
        reg_output = self.reg_core(reg_input).squeeze(-2)
        imagination_output, confidence = self.imagination(reg_output)

        logger.debug(
            "LEIBREG forward concluído text=%s image=%s memory=%s",
            text_point is not None,
            image_point is not None,
            memory_point is not None,
        )

        return {
            "text_point": text_point,
            "image_point": image_point,
            "memory_point": memory_point,
            "fused_point": fused_point,
            "reg_output": reg_output,
            "imagination_output": imagination_output,
            "imagination_confidence": confidence,
        }
