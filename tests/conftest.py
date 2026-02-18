"""Fixtures compartilhadas para todos os testes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import torch
from pyfolds.core import MPJRDConfig, MPJRDNeuron

@pytest.fixture
def small_config():
    """Configuração pequena para testes rápidos."""
    return MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        plastic=True
    )


@pytest.fixture
def full_config():
    """Configuração completa para testes de mecanismos avançados."""
    return MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        plastic=True
    )

@pytest.fixture
def tiny_config():
    """Configuração mínima para testes de unidade."""
    return MPJRDConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=2,
        plastic=True
    )

@pytest.fixture
def small_neuron(small_config):
    """Neurônio com configuração pequena."""
    return MPJRDNeuron(small_config)

@pytest.fixture
def device():
    """Device para testes (CPU sempre)."""
    return torch.device('cpu')

@pytest.fixture
def batch_size():
    """Batch size padrão para testes."""
    return 4