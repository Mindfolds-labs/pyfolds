"""Noetic bridge for LEIBREG multimodal geometric reasoning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pyfolds.advanced.engram import EngramBank
from pyfolds.leibreg.reg_core import REGCore
from pyfolds.leibreg.wave_context import extract_wave_phase
from pyfolds.leibreg.wordspace import WordSpace
from pyfolds.telemetry.types import TelemetryEvent


class NoeticLeibregBridge(nn.Module):
    """Project text/image/memory signals into LEIBREG conceptual space."""

    def __init__(
        self,
        *,
        dim_text: Optional[int] = None,
        dim_image: Optional[int] = None,
        dim_memory: Optional[int] = None,
        dim_concept: int = 4,
        concept_count: int = 1024,
        wordspace: Optional[WordSpace] = None,
        reg_core: Optional[REGCore] = None,
        engram_bank: Optional[EngramBank] = None,
        associative_memory: Optional[Any] = None,
        telemetry_collector: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.wordspace = wordspace or WordSpace(
            concept_count=concept_count,
            dim_base=dim_concept,
            text_dim=dim_text,
            image_dim=dim_image,
            memory_dim=dim_memory,
            wave_enabled=True,
            telemetry_collector=telemetry_collector,
        )
        self.reg_core = reg_core or REGCore(dim=dim_concept, telemetry_collector=telemetry_collector)
        self.engram_bank = engram_bank
        self.associative_memory = associative_memory
        self.telemetry_collector = telemetry_collector

    def _emit(self, payload: dict[str, float]) -> None:
        if self.telemetry_collector is None:
            return
        try:
            self.telemetry_collector.emit(TelemetryEvent(0.0, "leibreg_bridge", "noetic_pawp.leibreg_bridge", 0, payload))
        except Exception:
            return

    def _retrieve_memory(self, query: Any, concept_point: Tensor, top_k: int) -> Dict[str, Any]:
        if self.engram_bank is not None:
            found = self.engram_bank.search_by_resonance(query_pattern=concept_point.detach().float(), top_k=top_k)
            return {"backend": "engram_bank", "hits": len(found), "items": found}
        if self.associative_memory is not None:
            retriever = getattr(self.associative_memory, "retrieve", None)
            if callable(retriever):
                out = retriever(query)
                if isinstance(out, list):
                    return {"backend": "associative_memory", "hits": len(out), "items": out}
                return {"backend": "associative_memory", "hits": int(out is not None), "items": out}
            raise AttributeError("associative_memory must expose callable 'retrieve'")
        return {"backend": "missing", "hits": 0, "items": []}

    def forward(
        self,
        *,
        text_features: Optional[Tensor] = None,
        image_features: Optional[Tensor] = None,
        memory_features: Optional[Tensor] = None,
        concept_ids: Optional[Tensor] = None,
        wave_source: Optional[Any] = None,
        memory_query: Any = None,
        mask: Optional[Tensor] = None,
        top_k: int = 10,
        **legacy_kwargs: Any,
    ) -> Dict[str, Any]:
        if text_features is None and "texto" in legacy_kwargs:
            text_features = legacy_kwargs["texto"]
        if image_features is None and "imagem" in legacy_kwargs:
            image_features = legacy_kwargs["imagem"]

        points = []
        if text_features is not None:
            points.append(self.wordspace.project_text(text_features))
        if image_features is not None:
            points.append(self.wordspace.project_image(image_features))
        if memory_features is not None:
            points.append(self.wordspace.project_memory(memory_features))
        if concept_ids is not None:
            phase = extract_wave_phase(wave_source)
            ws_out = self.wordspace(concept_ids=concept_ids, phase=phase)
            points.append(ws_out["q_total"])
        if not points:
            raise ValueError("At least one modality or concept_ids must be provided")

        stacked = torch.stack(points, dim=0)
        concept_point = F.normalize(stacked.mean(dim=0), p=2, dim=-1)
        reg_input = concept_point.unsqueeze(1) if concept_point.ndim == 2 else concept_point
        reasoned = self.reg_core(reg_input, mask=mask)
        summary = F.normalize(reasoned.mean(dim=1), p=2, dim=-1)

        memory = self._retrieve_memory(memory_query, summary.flatten(), top_k)
        out = {
            "concept_point": summary,
            "activated_concepts": concept_ids,
            "memory_hits": memory["hits"],
            "memory": memory,
            "wave_metrics": {"phase_mean": float(extract_wave_phase(wave_source).mean().item()) if wave_source is not None and extract_wave_phase(wave_source) is not None else None},
        }
        self._emit(
            {
                "bridge_modalities_present": float(len(points)),
                "bridge_memory_hits": float(memory["hits"]),
                "bridge_output_norm": float(summary.norm(dim=-1).mean().item()),
            }
        )
        return out
