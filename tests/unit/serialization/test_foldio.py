from pyfolds.serialization.foldio import crc32c_u32
import numpy as np
import torch
import pytest

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import (
    FoldReader,
    ReedSolomonECC,
    ecc_from_protection,
    is_mind,
    load_fold_or_mind,
    peek_fold_or_mind,
    read_nuclear_arrays,
    save_fold_or_mind,
)

try:
    import reedsolo  # noqa: F401

    HAS_REEDSOLO = True
except Exception:
    HAS_REEDSOLO = False


def _build_neuron(enable_telemetry: bool = False) -> MPJRDNeuron:
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg, enable_telemetry=enable_telemetry, telemetry_profile="heavy")


def test_fold_roundtrip_and_peek(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "model.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        tags={"exp": "unit"},
        include_history=False,
        include_telemetry=False,
        compress="none",
    )

    peek = peek_fold_or_mind(str(file_path), use_mmap=True)

    assert "torch_state" in peek["chunks"]
    assert "llm_manifest" in peek["chunks"]
    assert "nuclear_arrays" in peek["chunks"]
    assert peek["metadata"]["model_type"] == "MPJRDNeuron"
    assert peek["is_mind"] is False
    assert is_mind(str(file_path)) is False

    loaded = load_fold_or_mind(str(file_path), MPJRDNeuron, map_location="cpu")

    assert loaded.__class__.__name__ == "MPJRDNeuron"
    assert loaded.cfg.n_dendrites == neuron.cfg.n_dendrites
    assert loaded.cfg.n_synapses_per_dendrite == neuron.cfg.n_synapses_per_dendrite


def test_training_then_save_with_telemetry_and_history_and_nuclear_arrays(tmp_path):
    neuron = _build_neuron(enable_telemetry=True)

    for _ in range(3):
        x = torch.rand(4, neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)
        neuron.forward(x, collect_stats=True)

    neuron.apply_plasticity(dt=1.0)
    file_path = tmp_path / "trained.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        tags={"phase": "trained"},
        include_history=True,
        include_telemetry=True,
        include_nuclear_arrays=True,
        compress="none",
    )

    info = peek_fold_or_mind(str(file_path), use_mmap=True)
    assert "metrics" in info["chunks"]
    assert "telemetry" in info["chunks"]
    assert "nuclear_arrays" in info["chunks"]
    assert info["llm_manifest"]["routing"]["resume_training"] == "torch_state"

    with FoldReader(str(file_path), use_mmap=True) as reader:
        telemetry = reader.read_json("telemetry")

    arrays = read_nuclear_arrays(str(file_path))
    assert telemetry["enabled"] is True
    assert isinstance(telemetry["events"], list)
    assert arrays["N"].shape == (neuron.cfg.n_dendrites, neuron.cfg.n_synapses_per_dendrite)
    assert np.isfinite(arrays["theta"]).all()


def test_detects_corruption(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "corrupt.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        torch_state_chunk = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        byte_to_flip = torch_state_chunk["offset"] + torch_state_chunk["header_len"] + 8

    with open(file_path, "r+b") as f:
        f.seek(byte_to_flip)
        original = f.read(1)
        f.seek(byte_to_flip)
        f.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(RuntimeError):
        with FoldReader(str(file_path), use_mmap=False) as reader:
            reader.read_chunk_bytes("torch_state", verify=True)


def test_ecc_from_protection_mapping():
    assert ecc_from_protection("off").name == "none"

    if HAS_REEDSOLO:
        assert ecc_from_protection("low").name == "rs(16)"
        assert ecc_from_protection("med").name == "rs(32)"
        assert ecc_from_protection("high").name == "rs(64)"
    else:
        with pytest.raises(ModuleNotFoundError):
            ecc_from_protection("low")

    with pytest.raises(ValueError):
        ecc_from_protection("ultra")


@pytest.mark.skipif(not HAS_REEDSOLO, reason="reedsolo não instalado")
def test_ecc_roundtrip_if_available(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "ecc.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=True,
        compress="none",
        ecc=ReedSolomonECC(symbols=16),
    )

    with FoldReader(str(file_path), use_mmap=True) as reader:
        chunks = {c["name"]: c for c in reader.index["chunks"]}

    assert chunks["torch_state"]["ecc_algo"] == "rs(16)"
    assert chunks["torch_state"]["ecc_len"] > 0


def test_crc32c_matches_known_vector():
    assert crc32c_u32(b"123456789") == 0xE3069283
