"""WordSpace: projeção multimodal para espaço conceitual comum 4D."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

DEFAULT_CONCEPT_DIM = 4


class WordSpace(nn.Module):
    """Projeta representações multimodais para um espaço conceitual comum.

    Cada modalidade possui adaptador próprio para evitar suposições sobre
    dimensionalidades equivalentes entre texto, imagem e memória.
    """

    _VALID_MODALITIES = {"text", "image", "memory"}

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        memory_dim: int,
        *,
        concept_dim: int = DEFAULT_CONCEPT_DIM,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if concept_dim <= 0:
            raise ValueError(f"concept_dim deve ser > 0, recebido: {concept_dim}")
        self.concept_dim = int(concept_dim)
        self.normalize = bool(normalize)

        self.text_adapter = self._make_adapter(text_dim, "text_dim")
        self.image_adapter = self._make_adapter(image_dim, "image_dim")
        self.memory_adapter = self._make_adapter(memory_dim, "memory_dim")

    def _make_adapter(self, input_dim: int, name: str) -> nn.Linear:
        if input_dim <= 0:
            raise ValueError(f"{name} deve ser > 0, recebido: {input_dim}")
        return nn.Linear(int(input_dim), self.concept_dim)

    def _validate_input(self, x: Tensor, *, expected_last_dim: int, modality: str) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(f"Entrada da modalidade '{modality}' deve ser torch.Tensor, recebido: {type(x)!r}")
        if x.ndim == 0:
            raise ValueError(f"Entrada da modalidade '{modality}' deve ter ao menos 1 dimensão.")
        if x.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Dimensão inválida para '{modality}': esperado último eixo={expected_last_dim}, recebido={x.shape[-1]}."
            )
        return x.float()

    def _post_process(self, x: Tensor) -> Tensor:
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-8)
        return x

    def _project(self, x: Tensor, adapter: nn.Linear, *, modality: str) -> Tensor:
        valid_x = self._validate_input(x, expected_last_dim=adapter.in_features, modality=modality)
        return self._post_process(adapter(valid_x))

    def project_text(self, x: Tensor) -> Tensor:
        """Projeta representação textual para o espaço conceitual."""
        return self._project(x, self.text_adapter, modality="text")

    def project_image(self, x: Tensor) -> Tensor:
        """Projeta representação visual para o espaço conceitual."""
        return self._project(x, self.image_adapter, modality="image")

    def project_memory(self, x: Tensor) -> Tensor:
        """Projeta representação de memória para o espaço conceitual."""
        return self._project(x, self.memory_adapter, modality="memory")

    def project(self, x: Tensor, *, modality: str) -> Tensor:
        """Projeta tensor de uma modalidade explícita para o espaço conceitual."""
        if modality not in self._VALID_MODALITIES:
            raise ValueError(f"Modalidade inválida '{modality}'. Opções: {sorted(self._VALID_MODALITIES)}")
        handlers: Mapping[str, nn.Module] = {
            "text": self.project_text,
            "image": self.project_image,
            "memory": self.project_memory,
        }
        fn = handlers[modality]
        return fn(x)  # type: ignore[operator]
