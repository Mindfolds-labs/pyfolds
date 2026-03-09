"""Sistema de engrams para memória distribuída no PyFolds."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EngramType(str, Enum):
    """Tipos semânticos suportados para um engram."""

    CONCEITO = "conceito"
    EPISODIO = "episodio"
    RELACAO = "relacao"
    HABILIDADE = "habilidade"
    REGRA = "regra"


@dataclass
class Engram:
    """Representa um traço de memória distribuído com metadados temporais."""

    signature: str
    concept: str
    engram_type: EngramType
    wave_pattern: torch.Tensor
    frequencies: torch.Tensor
    phases: torch.Tensor
    amplitudes: torch.Tensor
    formation_age: float
    circadian_phase: float
    meridiem: str
    day_of_life: int
    formation_timestamp: float
    importance: float = 0.5
    access_count: int = 0
    last_access: float = 0.0
    consolidated: bool = False
    relations: Dict[str, float] = field(default_factory=dict)
    area: str = "geral"
    tags: List[str] = field(default_factory=list)
    source: str = ""

    def __post_init__(self) -> None:
        """Normaliza tensores para CPU para serialização robusta."""
        self.wave_pattern = self.wave_pattern.detach().cpu().float()
        self.frequencies = self.frequencies.detach().cpu().float()
        self.phases = self.phases.detach().cpu().float()
        self.amplitudes = self.amplitudes.detach().cpu().float()

    def update_importance(self, delta: float) -> None:
        """Atualiza importância e registra reativação."""
        self.importance = float(max(0.0, min(1.0, self.importance + delta)))
        self.access_count += 1
        self.last_access = time.time()

    def similarity(self, other: "Engram") -> float:
        """Calcula similaridade por ressonância entre dois engrams."""
        freq_diff = torch.abs(self.frequencies - other.frequencies).mean()
        freq_sim = 1.0 - torch.sigmoid(freq_diff - 10.0)
        phase_diff = torch.abs(self.phases - other.phases)
        phase_diff = torch.minimum(phase_diff, 2 * math.pi - phase_diff)
        phase_sim = torch.cos(phase_diff).mean()
        amp_sim = F.cosine_similarity(self.amplitudes.unsqueeze(0), other.amplitudes.unsqueeze(0)).squeeze(0)
        return float((0.5 * freq_sim + 0.3 * phase_sim + 0.2 * amp_sim).item())

    def create_relation(self, other_signature: str, strength: float) -> None:
        """Cria ou atualiza relação com outro engram."""
        self.relations[other_signature] = float(max(0.0, min(1.0, strength)))

    def to_dict(self) -> Dict[str, Any]:
        """Converte metadados do engram para serialização JSON."""
        return {
            "signature": self.signature,
            "concept": self.concept,
            "type": self.engram_type.value,
            "formation_age": self.formation_age,
            "circadian_phase": self.circadian_phase,
            "meridiem": self.meridiem,
            "day_of_life": self.day_of_life,
            "formation_timestamp": self.formation_timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "consolidated": self.consolidated,
            "relations": self.relations,
            "area": self.area,
            "tags": self.tags,
            "source": self.source,
            "wave_pattern": self.wave_pattern.tolist(),
            "frequencies": self.frequencies.tolist(),
            "phases": self.phases.tolist(),
            "amplitudes": self.amplitudes.tolist(),
        }


class EngramBank(nn.Module):
    """Banco de memória por engrams com indexação e consolidação."""

    def __init__(
        self,
        max_engrams: int = 10_000_000,
        n_frequencies: int = 8,
        similarity_threshold: float = 0.7,
        pruning_threshold: float = 0.1,
        enable_indexing: bool = True,
        eviction_strategy: Literal["importance", "lru", "lru_importance"] = "importance",
    ) -> None:
        """Inicializa capacidade, índices e buffers de estatísticas."""
        super().__init__()
        self.max_engrams = max_engrams
        self.n_frequencies = n_frequencies
        self.similarity_threshold = similarity_threshold
        self.pruning_threshold = pruning_threshold
        self.enable_indexing = enable_indexing
        self.eviction_strategy = eviction_strategy
        self.engrams: Dict[str, Engram] = {}
        self.area_index: Dict[str, List[str]] = defaultdict(list)
        self.phase_index: Dict[int, List[str]] = defaultdict(list)
        self.freq_index: Dict[int, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.search_cache: Dict[str, List[Engram]] = {}
        self.register_buffer("total_engrams", torch.tensor(0, dtype=torch.long))
        self.register_buffer("consolidated_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_pruning", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_rng_seed", torch.tensor(7, dtype=torch.long))
        self.cache_hits = 0
        self.cache_misses = 0

    def create_engram(
        self,
        wave_pattern: torch.Tensor,
        concept: str,
        age: float,
        phase: float,
        meridiem: str,
        engram_type: EngramType = EngramType.CONCEITO,
        importance: float = 0.5,
        area: str = "geral",
        tags: Optional[List[str]] = None,
        source: str = "",
    ) -> Engram:
        """Cria e indexa um novo engram."""
        wave_pattern = wave_pattern.detach().float().to(self.total_engrams.device).flatten()
        if wave_pattern.numel() < self.n_frequencies:
            wave_pattern = F.pad(wave_pattern, (0, self.n_frequencies - wave_pattern.numel()))
        elif wave_pattern.numel() > self.n_frequencies:
            wave_pattern = wave_pattern[: self.n_frequencies]

        # assinatura é usada para indexação/identificação, não para segurança criptográfica
        pattern_int = int(wave_pattern[:4].sum().item() * 1e6) & 0xFFFFFFFF
        timestamp = time.time()
        signature = f"{pattern_int:08x}_{int(age)}_{int(phase)}_{int(timestamp*1e6)}_{int(self.total_engrams.item())}"
        freqs = torch.linspace(10.0, 100.0, self.n_frequencies, device=wave_pattern.device)
        gen = torch.Generator(device=wave_pattern.device)
        gen.manual_seed(int(self._rng_seed.item()) + int(self.total_engrams.item()))
        phases = torch.rand(self.n_frequencies, generator=gen, device=wave_pattern.device) * (2 * math.pi)

        engram = Engram(
            signature=signature,
            concept=concept,
            engram_type=engram_type,
            wave_pattern=wave_pattern,
            frequencies=freqs,
            phases=phases,
            amplitudes=wave_pattern.clone(),
            formation_age=float(age),
            circadian_phase=float(phase),
            meridiem=meridiem,
            day_of_life=int(age / (24 * 3600)),
            formation_timestamp=timestamp,
            importance=float(importance),
            area=area,
            tags=tags or [],
            source=source,
        )
        self.engrams[signature] = engram
        self.total_engrams.add_(1)
        self._index_engram(engram)
        if len(self.engrams) > self.max_engrams:
            self._prune_oldest()
        return engram

    def _index_engram(self, engram: Engram) -> None:
        """Atualiza índices secundários."""
        if not self.enable_indexing:
            return
        self.area_index[engram.area].append(engram.signature)
        self.phase_index[int(engram.circadian_phase / 30) * 30].append(engram.signature)
        self.freq_index[int(float(engram.frequencies.mean().item()) / 10) * 10].append(engram.signature)
        for tag in engram.tags:
            self.tag_index[tag].append(engram.signature)

    def search_by_resonance(
        self,
        query_pattern: torch.Tensor,
        query_phase: Optional[float] = None,
        area: Optional[str] = None,
        top_k: int = 10,
        use_cache: bool = True,
    ) -> List[Engram]:
        """Busca engrams por ressonância de padrão, fase e importância."""
        query_pattern = query_pattern.detach().float().flatten().cpu()
        if query_pattern.numel() < self.n_frequencies:
            query_pattern = F.pad(query_pattern, (0, self.n_frequencies - query_pattern.numel()))
        elif query_pattern.numel() > self.n_frequencies:
            query_pattern = query_pattern[: self.n_frequencies]

        cache_key = f"{hash(query_pattern.numpy().tobytes())}_{query_phase}_{area}_{top_k}"
        if use_cache and cache_key in self.search_cache:
            self.cache_hits += 1
            return self.search_cache[cache_key]
        self.cache_misses += 1

        candidates = [self.engrams[s] for s in self.area_index.get(area, [])] if area else list(self.engrams.values())
        scores: List[Tuple[Engram, float]] = []
        for engram in candidates:
            pattern_sim = float(F.cosine_similarity(query_pattern.unsqueeze(0), engram.wave_pattern.unsqueeze(0)).item())
            if pattern_sim < 0.3:
                continue
            phase_factor = 1.0
            if query_phase is not None:
                pd = abs(engram.circadian_phase - float(query_phase))
                pd = min(pd, 360.0 - pd)
                phase_factor = max(0.0, math.cos(math.radians(pd)) ** 2)
            resonance = pattern_sim * (0.6 + 0.2 * phase_factor + 0.2 * (engram.importance**2))
            scores.append((engram, resonance))

        scores.sort(key=lambda item: item[1], reverse=True)
        out = [item[0] for item in scores[:top_k]]
        if use_cache:
            self.search_cache[cache_key] = out
            if len(self.search_cache) > 1000:
                self.search_cache.pop(next(iter(self.search_cache)))
        return out

    def create_relation(self, sig1: str, sig2: str, strength: float) -> None:
        """Cria ligação bidirecional entre engrams existentes."""
        if sig1 in self.engrams and sig2 in self.engrams:
            self.engrams[sig1].create_relation(sig2, strength)
            self.engrams[sig2].create_relation(sig1, strength)

    def get_by_time(self, target_age: float, window: float = 3600, area: Optional[str] = None) -> List[Engram]:
        """Recupera engrams próximos a uma idade alvo."""
        candidates = [self.engrams[s] for s in self.area_index.get(area, [])] if area else list(self.engrams.values())
        out = [e for e in candidates if abs(e.formation_age - target_age) <= window]
        out.sort(key=lambda e: abs(e.formation_age - target_age))
        return out

    def replay(self, batch_size: int = 32) -> List[Engram]:
        """Reativa lote de memórias importantes para reforço durante sono."""
        if not self.engrams:
            return []
        now = time.time()
        ranked: List[Tuple[float, Engram]] = []
        for e in self.engrams.values():
            recency = 1.0 / (1.0 + (now - e.last_access) / 3600.0)
            ranked.append((e.importance * recency, e))
        ranked.sort(key=lambda x: x[0], reverse=True)
        replayed = [e for _, e in ranked[:batch_size]]
        for e in replayed:
            e.update_importance(0.05)
        logger.info("replay_batch=%s", len(replayed))
        return replayed

    def consolidate(self, pruning: bool = True) -> Dict[str, int]:
        """Marca memórias importantes e executa pruning opcional."""
        stats = {"before": len(self.engrams), "consolidated": 0, "pruned": 0}
        for e in self.engrams.values():
            if e.importance > 0.7 and not e.consolidated:
                e.consolidated = True
                self.consolidated_count.add_(1)
                stats["consolidated"] += 1

        if pruning:
            dead = [sig for sig, e in self.engrams.items() if e.importance < self.pruning_threshold]
            for sig in dead:
                del self.engrams[sig]
            stats["pruned"] = len(dead)
            self._rebuild_indexes()
            logger.info("pruning removed=%s threshold=%.3f", len(dead), self.pruning_threshold)

        stats["after"] = len(self.engrams)
        self.last_pruning.fill_(time.time())
        self.total_engrams.fill_(len(self.engrams))
        return stats

    def _rebuild_indexes(self) -> None:
        """Reconstrói índices secundários após remoções em lote."""
        self.area_index.clear()
        self.phase_index.clear()
        self.freq_index.clear()
        self.tag_index.clear()
        for e in self.engrams.values():
            self._index_engram(e)

    def _prune_oldest(self) -> None:
        """Poda 10% dos engrams conforme estratégia de eviction configurada."""
        if len(self.engrams) <= self.max_engrams:
            return
        now = time.time()

        def score(sig: str, e: Engram) -> float:
            if self.eviction_strategy == "lru":
                return float(e.last_access)
            if self.eviction_strategy == "lru_importance":
                recency_score = 1.0 / (1.0 + (now - e.last_access) / 3600.0)
                access_frequency = min(1.0, e.access_count / 100.0)
                return (e.importance * 0.6) + (recency_score * 0.3) + (access_frequency * 0.1)
            return (e.importance * 1.0) + (e.formation_age * 1e-6)

        ranked = sorted(((score(sig, e), sig) for sig, e in self.engrams.items()), key=lambda x: x[0])
        for _, sig in ranked[: max(1, int(0.1 * len(self.engrams)))]:
            del self.engrams[sig]
        self._rebuild_indexes()
        self.total_engrams.fill_(len(self.engrams))

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas operacionais do banco de memória."""
        total_q = self.cache_hits + self.cache_misses
        return {
            "total_engrams": len(self.engrams),
            "max_engrams": self.max_engrams,
            "consolidated": int(self.consolidated_count.item()),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": (self.cache_hits / total_q) if total_q else 0.0,
        }

    def save_state(self) -> Dict[str, Any]:
        """Serializa banco de engrams para persistência."""
        return {
            "engrams": {sig: e.to_dict() for sig, e in self.engrams.items()},
            "config": {
                "max_engrams": self.max_engrams,
                "n_frequencies": self.n_frequencies,
                "similarity_threshold": self.similarity_threshold,
                "pruning_threshold": self.pruning_threshold,
                "eviction_strategy": self.eviction_strategy,
            },
            "stats": self.get_stats(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restaura banco a partir de dicionário serializado."""
        self.engrams.clear()
        cfg = state.get("config", {})
        self.max_engrams = int(cfg.get("max_engrams", self.max_engrams))
        self.n_frequencies = int(cfg.get("n_frequencies", self.n_frequencies))
        self.similarity_threshold = float(cfg.get("similarity_threshold", self.similarity_threshold))
        self.pruning_threshold = float(cfg.get("pruning_threshold", self.pruning_threshold))
        self.eviction_strategy = str(cfg.get("eviction_strategy", self.eviction_strategy))

        for sig, data in state.get("engrams", {}).items():
            engram = Engram(
                signature=sig,
                concept=data["concept"],
                engram_type=EngramType(data.get("type", EngramType.CONCEITO.value)),
                wave_pattern=torch.tensor(data.get("wave_pattern", [0.0] * self.n_frequencies), dtype=torch.float32),
                frequencies=torch.tensor(data.get("frequencies", [0.0] * self.n_frequencies), dtype=torch.float32),
                phases=torch.tensor(data.get("phases", [0.0] * self.n_frequencies), dtype=torch.float32),
                amplitudes=torch.tensor(data.get("amplitudes", [0.0] * self.n_frequencies), dtype=torch.float32),
                formation_age=float(data.get("formation_age", 0.0)),
                circadian_phase=float(data.get("circadian_phase", 0.0)),
                meridiem=str(data.get("meridiem", "AM")),
                day_of_life=int(data.get("day_of_life", 0)),
                formation_timestamp=float(data.get("formation_timestamp", 0.0)),
                importance=float(data.get("importance", 0.5)),
                access_count=int(data.get("access_count", 0)),
                last_access=float(data.get("last_access", 0.0)),
                consolidated=bool(data.get("consolidated", False)),
                relations=dict(data.get("relations", {})),
                area=str(data.get("area", "geral")),
                tags=list(data.get("tags", [])),
                source=str(data.get("source", "")),
            )
            self.engrams[sig] = engram

        self.total_engrams.fill_(len(self.engrams))
        self.consolidated_count.fill_(sum(1 for e in self.engrams.values() if e.consolidated))
        self._rebuild_indexes()
