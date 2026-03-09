"""Testes de regressão para o modelo noético."""

import torch

from pyfolds.advanced.engram import EngramBank
from pyfolds.advanced.noetic_model import NoeticCore
from pyfolds.core.config import MPJRDConfig


def test_engram_creation() -> None:
    """Valida criação de engram a partir do ciclo de aprendizado."""
    noetic = NoeticCore(MPJRDConfig())
    engram = noetic.learn("teste", importance=0.8)
    assert engram.concept == "teste"
    assert engram.importance == 0.8
    assert int(noetic.engram_bank.total_engrams.item()) == 1


def test_search_by_resonance() -> None:
    """Valida busca por ressonância com filtro de área."""
    noetic = NoeticCore(MPJRDConfig())
    noetic.learn("gato", area="biologia", pattern=torch.ones(noetic.engram_bank.n_frequencies))
    noetic.learn("cachorro", area="biologia", pattern=torch.ones(noetic.engram_bank.n_frequencies) * 0.9)
    noetic.learn("fisica", area="fisica", pattern=torch.ones(noetic.engram_bank.n_frequencies) * 0.1)
    results = noetic.query(torch.ones(noetic.engram_bank.n_frequencies), area="biologia", top_k=2)
    assert len(results) > 0
    assert results[0]["area"] == "biologia"


def test_sleep_consolidation() -> None:
    """Valida pruning de memórias fracas durante sono."""
    noetic = NoeticCore(MPJRDConfig(pruning_threshold=0.5))
    noetic.learn("importante", importance=0.9)
    noetic.learn("medio", importance=0.5)
    noetic.learn("fraco", importance=0.3)
    noetic.sleep()
    assert int(noetic.engram_bank.total_engrams.item()) == 2


def test_specialization() -> None:
    """Valida criação de conceitos especializados em profundidade."""
    noetic = NoeticCore(MPJRDConfig())
    noetic.specialization.define_area("fisica", "Física", base_frequency=50.0)
    created = noetic.specialization.specialize("fisica", "gravidade", depth=2)
    assert len(created) >= 2


def test_save_load(tmp_path) -> None:
    """Valida persistência de estado noético."""
    cfg = MPJRDConfig()
    noetic = NoeticCore(cfg)
    noetic.learn("teste_save", importance=0.9)
    target = tmp_path / "teste_noetic.mind"
    noetic.save(str(target))
    loaded = NoeticCore.load(str(target), cfg)
    assert int(loaded.engram_bank.total_engrams.item()) == 1


def test_engram_bank_isolated() -> None:
    """Exercita banco de engrams isoladamente."""
    bank = EngramBank(max_engrams=10, n_frequencies=4)
    e1 = bank.create_engram(torch.ones(4), "a", age=0.0, phase=10.0, meridiem="AM")
    bank.create_engram(torch.ones(4) * 0.9, "b", age=1.0, phase=20.0, meridiem="AM")
    res = bank.search_by_resonance(torch.ones(4), area=None, top_k=1)
    assert res[0].signature == e1.signature
