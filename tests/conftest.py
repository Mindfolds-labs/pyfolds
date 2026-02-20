"""Fixtures compartilhadas para todos os testes."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

TORCH_AVAILABLE = find_spec("torch") is not None
TORCH_REASON = (
    "PyTorch não instalado no ambiente de testes. "
    "Para suíte completa rode: pip install -e .[dev]"
)


@lru_cache(maxsize=1)
def _module_requires_torch(path: Path) -> bool:
    """Define se um módulo de teste depende de PyTorch."""
    normalized = path.as_posix()
    if "/tests/tools/" in normalized:
        return False

    content = path.read_text(encoding="utf-8")
    torch_signals = (
        "import torch",
        "from torch",
        "pytest.importorskip(\"torch\")",
        "pytest.importorskip('torch')",
        "from pyfolds.core",
        "from pyfolds.network",
    )
    return any(signal in content for signal in torch_signals)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "torch_required: marca testes que exigem PyTorch",
    )


def pytest_report_header(config: pytest.Config) -> list[str]:
    if TORCH_AVAILABLE:
        return ["preflight: torch detectado"]
    return [f"preflight: {TORCH_REASON}"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if TORCH_AVAILABLE:
        return

    skip_no_torch = pytest.mark.skip(reason=TORCH_REASON)
    for item in items:
        if _module_requires_torch(Path(str(item.fspath))):
            item.add_marker(skip_no_torch)
            item.add_marker("torch_required")


@pytest.fixture
def torch_module():
    """Módulo torch disponível para testes dependentes de backend."""
    return pytest.importorskip("torch", reason=TORCH_REASON)


@pytest.fixture
def _core_symbols():
    """Carrega símbolos de core apenas quando torch está disponível."""
    pytest.importorskip("torch", reason=TORCH_REASON)
    from pyfolds import NeuronConfig, MPJRDNeuron

    return NeuronConfig, MPJRDNeuron


@pytest.fixture
def small_config(_core_symbols):
    """Configuração pequena para testes rápidos."""
    NeuronConfig, _ = _core_symbols
    return NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        plastic=True,
    )


@pytest.fixture
def full_config(_core_symbols):
    """Configuração completa para testes de mecanismos avançados."""
    NeuronConfig, _ = _core_symbols
    return NeuronConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=8,
        plastic=True,
    )


@pytest.fixture
def tiny_config(_core_symbols):
    """Configuração mínima para testes de unidade."""
    NeuronConfig, _ = _core_symbols
    return NeuronConfig(
        n_dendrites=1,
        n_synapses_per_dendrite=2,
        plastic=True,
    )


@pytest.fixture
def small_neuron(small_config, _core_symbols):
    """Neurônio com configuração pequena."""
    _, MPJRDNeuron = _core_symbols
    return MPJRDNeuron(small_config)


@pytest.fixture
def device(torch_module):
    """Device para testes (CPU sempre)."""
    return torch_module.device("cpu")


@pytest.fixture
def batch_size():
    """Batch size padrão para testes."""
    return 4
