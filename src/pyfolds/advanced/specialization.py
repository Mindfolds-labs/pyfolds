"""Sistema de especialização e síntese interdisciplinar para engrams."""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .engram import Engram, EngramBank, EngramType

logger = logging.getLogger(__name__)


class KnowledgeHierarchy:
    """Organiza engrams por área e nível de abstração."""

    def __init__(self) -> None:
        """Inicializa estrutura área->nível->assinaturas."""
        self.areas: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))

    def add_to_hierarchy(self, engram: Engram, level: int) -> None:
        """Adiciona engram em um nível da hierarquia."""
        self.areas[engram.area][level].append(engram.signature)

    def get_at_level(self, area: str, level: int) -> List[str]:
        """Retorna assinaturas de um nível específico da área."""
        return self.areas.get(area, {}).get(level, [])

    def get_abstraction_chain(self, signature: str, engram_bank: EngramBank) -> List[Engram]:
        """Obtém cadeia em direção a conceitos mais abstratos."""
        if signature not in engram_bank.engrams:
            return []
        current = engram_bank.engrams[signature]
        chain = [current]
        for _ in range(4):
            target_freq = float(current.frequencies.mean().item()) * 2
            bucket = engram_bank.freq_index.get(int(target_freq / 10) * 10, [])
            nxt = next((engram_bank.engrams[s] for s in bucket if s in engram_bank.engrams and engram_bank.engrams[s].area == current.area and s != current.signature), None)
            if nxt is None:
                break
            chain.append(nxt)
            current = nxt
        return chain

    def get_specialization_chain(self, signature: str, engram_bank: EngramBank) -> List[Engram]:
        """Obtém cadeia em direção a conceitos mais específicos."""
        if signature not in engram_bank.engrams:
            return []
        current = engram_bank.engrams[signature]
        chain = [current]
        for _ in range(4):
            target_freq = float(current.frequencies.mean().item()) / 2
            bucket = engram_bank.freq_index.get(int(target_freq / 10) * 10, [])
            nxt = next((engram_bank.engrams[s] for s in bucket if s in engram_bank.engrams and engram_bank.engrams[s].area == current.area and s != current.signature), None)
            if nxt is None:
                break
            chain.append(nxt)
            current = nxt
        return chain


class SpecializationEngine(nn.Module):
    """Gerencia áreas de conhecimento, profundidade e síntese entre áreas."""

    def __init__(self, engram_bank: EngramBank) -> None:
        """Cria motor de especialização sobre um banco de engrams."""
        super().__init__()
        self.engram_bank = engram_bank
        self.hierarchy = KnowledgeHierarchy()
        self.areas: Dict[str, Dict[str, object]] = {}
        self.synthesis: List[Dict[str, object]] = []

    def define_area(
        self,
        name: str,
        description: str,
        parent_area: Optional[str] = None,
        base_frequency: float = 50.0,
    ) -> None:
        """Registra área de conhecimento e parâmetros de base."""
        self.areas[name] = {
            "name": name,
            "description": description,
            "parent": parent_area,
            "base_frequency": float(base_frequency),
            "created_at": time.time(),
            "n_concepts": 0,
        }

    def specialize(self, area: str, concept: str, depth: int = 3) -> List[Engram]:
        """Cria engrams especializados para um conceito em uma área."""
        if area not in self.areas:
            raise ValueError(f"Área {area} não definida")

        created: List[Engram] = []
        base_freq = float(self.areas[area]["base_frequency"])
        query = torch.ones(self.engram_bank.n_frequencies) * base_freq
        base_hits = self.engram_bank.search_by_resonance(query_pattern=query, area=area, top_k=1)

        if base_hits:
            base = base_hits[0]
        else:
            base = self.engram_bank.create_engram(
                wave_pattern=query,
                concept=concept,
                age=float(self.engram_bank.last_pruning.item()),
                phase=0.0,
                meridiem="AM",
                engram_type=EngramType.CONCEITO,
                importance=0.8,
                area=area,
            )
            created.append(base)

        for level in range(1, depth + 1):
            spec_freq = base_freq / (2**level)
            spec = self.engram_bank.create_engram(
                wave_pattern=torch.ones(self.engram_bank.n_frequencies) * spec_freq,
                concept=f"{concept}_level{level}",
                age=float(self.engram_bank.last_pruning.item()),
                phase=0.0,
                meridiem="AM",
                engram_type=EngramType.CONCEITO,
                importance=0.9,
                area=area,
                tags=[area, concept, f"level_{level}"],
            )
            self.engram_bank.create_relation(base.signature, spec.signature, strength=0.8 / level)
            self.hierarchy.add_to_hierarchy(spec, level)
            created.append(spec)

        self.areas[area]["n_concepts"] = int(self.areas[area]["n_concepts"]) + len(created)
        return created

    def synthesize(self, area1: str, area2: str) -> Optional[Engram]:
        """Tenta criar um engram de síntese com base na ressonância entre áreas."""
        if area1 not in self.areas or area2 not in self.areas:
            return None
        q1 = torch.ones(self.engram_bank.n_frequencies) * float(self.areas[area1]["base_frequency"])
        q2 = torch.ones(self.engram_bank.n_frequencies) * float(self.areas[area2]["base_frequency"])
        concepts1 = self.engram_bank.search_by_resonance(query_pattern=q1, area=area1, top_k=10)
        concepts2 = self.engram_bank.search_by_resonance(query_pattern=q2, area=area2, top_k=10)

        best_res = 0.0
        best_pair: Optional[tuple[Engram, Engram]] = None
        for c1 in concepts1:
            for c2 in concepts2:
                sim = c1.similarity(c2)
                if sim > best_res:
                    best_res = sim
                    best_pair = (c1, c2)

        if best_pair is None or best_res <= 0.6:
            return None

        c1, c2 = best_pair
        synth_freq = (c1.frequencies.mean() + c2.frequencies.mean()) / 2.0
        synthesis = self.engram_bank.create_engram(
            wave_pattern=torch.ones(self.engram_bank.n_frequencies) * float(synth_freq.item()),
            concept=f"{area1}_{area2}_{c1.concept}_{c2.concept}",
            age=float(self.engram_bank.last_pruning.item()),
            phase=0.0,
            meridiem="AM",
            engram_type=EngramType.RELACAO,
            importance=1.0,
            area=f"{area1}_{area2}",
            tags=[area1, area2, "synthesis"],
        )
        self.engram_bank.create_relation(synthesis.signature, c1.signature, 0.9)
        self.engram_bank.create_relation(synthesis.signature, c2.signature, 0.9)
        event = {
            "area1": area1,
            "area2": area2,
            "concept1": c1.concept,
            "concept2": c2.concept,
            "synthesis": synthesis.concept,
            "resonance": best_res,
            "timestamp": time.time(),
        }
        self.synthesis.append(event)
        logger.info("synthesis_created area1=%s area2=%s resonance=%.3f", area1, area2, best_res)
        return synthesis

    def get_area_summary(self, area: str) -> Dict[str, object]:
        """Retorna resumo estatístico da área."""
        if area not in self.areas:
            return {}
        area_engrams = [self.engram_bank.engrams[s] for s in self.engram_bank.area_index.get(area, []) if s in self.engram_bank.engrams]
        if not area_engrams:
            return dict(self.areas[area])
        levels: Dict[int, int] = defaultdict(int)
        base_frequency = float(self.areas[area]["base_frequency"])
        for e in area_engrams:
            freq = max(1e-6, float(e.frequencies.mean().item()))
            levels[int(math.log2(base_frequency / freq))] += 1
        return {
            **self.areas[area],
            "n_engrams": len(area_engrams),
            "mean_importance": sum(e.importance for e in area_engrams) / len(area_engrams),
            "levels": dict(levels),
        }

    def get_synthesis_summary(self) -> List[Dict[str, object]]:
        """Retorna as sínteses mais recentes."""
        return self.synthesis[-20:]
