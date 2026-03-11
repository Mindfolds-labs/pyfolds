"""Modelo noético com memória por engrams e especialização."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union

import torch

from ..core.config import MPJRDConfig
from ..core.neuron import MPJRDNeuron
from ..utils.types import LearningMode
from .circadian import CircadianWaveMixin
from .engram import Engram, EngramBank, EngramType
from .specialization import SpecializationEngine
from .wave import WaveMixin

logger = logging.getLogger(__name__)


class NoeticCore(CircadianWaveMixin, WaveMixin, MPJRDNeuron):
    """Núcleo cognitivo que integra tempo, ondas e memória distribuída."""

    def __init__(self, cfg: MPJRDConfig):
        """Inicializa o modelo noético com bancos e contadores."""
        super().__init__(cfg)
        self._init_wave(cfg)
        self._init_circadian(cfg)
        self.register_buffer("birth_time", torch.tensor(time.time(), dtype=torch.float32))
        self.register_buffer("total_experiences", torch.tensor(0, dtype=torch.long))
        self.register_buffer("sleep_cycles", torch.tensor(0, dtype=torch.long))
        self.register_buffer("discoveries", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_name_hash", torch.tensor(0, dtype=torch.long))
        self.mode = LearningMode.ONLINE

        self.engram_bank = EngramBank(
            max_engrams=int(getattr(cfg, "max_engrams", 10_000_000)),
            n_frequencies=int(getattr(cfg, "engram_n_frequencies", cfg.wave_n_frequencies)),
            pruning_threshold=float(getattr(cfg, "pruning_threshold", 0.1)),
        )
        self.specialization = SpecializationEngine(self.engram_bank)

    def get_current_context(self) -> Dict[str, Any]:
        """Retorna contexto temporal atual para formação de engrams."""
        if getattr(self, "_circadian_enabled", False):
            return self._get_circadian_context()
        age = float(getattr(self, "age_seconds", torch.tensor(0.0)).item())
        phase = (age % (12 * 3600)) * 360.0 / (12 * 3600)
        return {"phase": phase, "meridiem": "AM" if phase < 180 else "PM", "day": int(age // (24 * 3600))}

    def advance_age(self, dt: float) -> None:
        """Avança idade do modelo em segundos."""
        if getattr(self, "_circadian_enabled", False):
            self._advance_circadian(dt)
        elif hasattr(self, "age_seconds"):
            self.age_seconds.add_(float(dt))

    def learn(
        self,
        concept: str,
        pattern: Optional[torch.Tensor] = None,
        area: str = "geral",
        importance: float = 0.5,
        source: str = "",
    ) -> Engram:
        """Aprende um novo conceito criando um engram."""
        if pattern is None:
            base_freq = float(self.specialization.areas.get(area, {}).get("base_frequency", 50.0))
            pattern = torch.ones(self.engram_bank.n_frequencies, device=self.theta.device) * base_freq
        else:
            pattern = pattern.to(self.theta.device)

        ctx = self.get_current_context()
        engram = self.engram_bank.create_engram(
            wave_pattern=pattern,
            concept=concept,
            age=float(getattr(self, "age_seconds", torch.tensor(0.0)).item()),
            phase=float(ctx["phase"]),
            meridiem=str(ctx["meridiem"]),
            engram_type=EngramType.CONCEITO,
            importance=importance,
            area=area,
            source=source,
        )
        self.total_experiences.add_(1)
        return engram

    def query(self, query: Union[str, torch.Tensor], area: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """Consulta memória por ressonância."""
        if isinstance(query, str):
            digest = hashlib.sha256(query.encode("utf-8")).digest()
            vals = [digest[i] / 255.0 for i in range(self.engram_bank.n_frequencies)]
            pattern = torch.tensor(vals, dtype=torch.float32)
        else:
            pattern = query.detach().float()
        ctx = self.get_current_context()
        matches = self.engram_bank.search_by_resonance(pattern, query_phase=float(ctx["phase"]), area=area, top_k=top_k)
        return [
            {
                "concept": e.concept,
                "area": e.area,
                "importance": e.importance,
                "age": e.formation_age,
                "phase": e.circadian_phase,
                "meridiem": e.meridiem,
                "signature": e.signature,
            }
            for e in matches
        ]

    def remember_when(self, concept: str) -> List[float]:
        """Retorna idades de formação para um conceito."""
        return sorted(e.formation_age for e in self.engram_bank.engrams.values() if e.concept == concept)

    def sleep(self, duration: Optional[float] = None) -> None:
        """Executa ciclo de sono com replay, pruning e síntese."""
        logger.info("sleep_start age=%s", self.get_age_string())
        self.mode = LearningMode.SLEEP
        self.neuromodulator.set_mode(LearningMode.SLEEP)
        self.engram_bank.replay(batch_size=int(getattr(self.cfg, "replay_batch_size", 32)))
        self.engram_bank.consolidate(pruning=True)

        areas = list(self.specialization.areas.keys())
        if len(areas) >= 2 and int(self.sleep_cycles.item()) % 7 == 0:
            synth = self.specialization.synthesize(areas[0], areas[1])
            if synth is not None:
                self.discoveries.add_(1)

        self.sleep_cycles.add_(1)
        logger.info("sleep_end cycles=%s", int(self.sleep_cycles.item()))

    def get_age_string(self) -> str:
        """Formata idade como dias e horas."""
        age_seconds = float(getattr(self, "age_seconds", torch.tensor(0.0)).item())
        days = int(age_seconds / (24 * 3600))
        hours = int((age_seconds % (24 * 3600)) / 3600)
        return f"{days}d {hours}h" if days else f"{hours}h"

    def get_life_story(self) -> str:
        """Retorna resumo textual da trajetória do modelo."""
        stats = self.engram_bank.get_stats()
        return (
            f"Noetic age={self.get_age_string()} memories={stats['total_engrams']} "
            f"sleep_cycles={int(self.sleep_cycles.item())} discoveries={int(self.discoveries.item())}"
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        """Forward padrão com suporte opcional a aprendizado/consulta."""
        dt = float(kwargs.get("dt", 0.0))
        if dt > 0:
            self.advance_age(dt)
        out = super().forward(x, **kwargs)
        if kwargs.get("learn", False) and "concept" in kwargs:
            pattern = torch.tensor(out.get("wave_frequencies", []), dtype=torch.float32, device=self.theta.device)
            engram = self.learn(kwargs["concept"], pattern=pattern, area=kwargs.get("area", "geral"), importance=float(kwargs.get("importance", 0.5)))
            out["learned"] = engram.signature
        if kwargs.get("query", False):
            out["query_results"] = self.query(torch.tensor(out.get("wave_frequencies", []), dtype=torch.float32), area=kwargs.get("area"), top_k=int(kwargs.get("top_k", 10)))
        out["age_seconds"] = float(getattr(self, "age_seconds", torch.tensor(0.0)).item())
        out["total_memories"] = int(self.engram_bank.total_engrams.item())
        return out


    def collect_engram_report(self, top_k: int = 10) -> Dict[str, Any]:
        """Exporta métricas observáveis de memória e ressonância por engrama."""
        base = super().collect_engram_report()
        ranked = sorted(
            self.engram_bank.engrams.values(),
            key=lambda e: (float(e.importance), float(e.access_count)),
            reverse=True,
        )[: max(1, int(top_k))]
        base.update({
            "engram_count": int(len(self.engram_bank.engrams)),
            "top_engrams": [
                {
                    "signature": e.signature,
                    "concept": e.concept,
                    "importance": float(e.importance),
                    "access_count": int(e.access_count),
                    "consolidated": bool(e.consolidated),
                }
                for e in ranked
            ],
        })
        return base

    def save(self, path: str, private_key: Optional[str] = None) -> None:
        """Salva estado noético para arquivo local."""
        _ = private_key
        payload = {
            "state_dict": self.state_dict(),
            "engram_bank": self.engram_bank.save_state(),
            "specialization": {"areas": self.specialization.areas, "synthesis": self.specialization.synthesis},
            "meta": {
                "sleep_cycles": int(self.sleep_cycles.item()),
                "discoveries": int(self.discoveries.item()),
                "total_experiences": int(self.total_experiences.item()),
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, cfg: MPJRDConfig, public_key: Optional[str] = None) -> "NoeticCore":
        """Carrega estado noético previamente salvo."""
        _ = public_key
        payload = torch.load(path, map_location="cpu")
        model = cls(cfg)
        model.load_state_dict(payload["state_dict"], strict=False)
        model.engram_bank.load_state(payload.get("engram_bank", {}))
        spec = payload.get("specialization", {})
        model.specialization.areas = spec.get("areas", {})
        model.specialization.synthesis = spec.get("synthesis", [])
        meta = payload.get("meta", {})
        model.sleep_cycles.fill_(int(meta.get("sleep_cycles", 0)))
        model.discoveries.fill_(int(meta.get("discoveries", 0)))
        model.total_experiences.fill_(int(meta.get("total_experiences", 0)))
        return model
