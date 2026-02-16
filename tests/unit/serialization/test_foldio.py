import mmap
import struct

import numpy as np
import pytest
import torch

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
from pyfolds.serialization.foldio import HEADER_FMT, MAGIC, MAX_CHUNK_SIZE, FoldWriter, crc32c_u32


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


def test_hierarchical_hashes_present_in_metadata(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "hashes.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        metadata = reader.index.get("metadata", {})

    assert "chunk_hashes" in metadata
    assert "manifest_hash" in metadata
    assert "torch_state" in metadata["chunk_hashes"]


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


def test_ecc_roundtrip_if_available(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "ecc.fold"

    ecc_codec = ReedSolomonECC(symbols=16) if HAS_REEDSOLO else ecc_from_protection("off")

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=True,
        compress="none",
        ecc=ecc_codec,
    )

    with FoldReader(str(file_path), use_mmap=True) as reader:
        chunks = {c["name"]: c for c in reader.index["chunks"]}

    if HAS_REEDSOLO:
        assert chunks["torch_state"]["ecc_algo"] == "rs(16)"
        assert chunks["torch_state"]["ecc_len"] > 0
    else:
        assert chunks["torch_state"]["ecc_algo"] == "none"
        assert chunks["torch_state"]["ecc_len"] == 0


def test_crc32c_matches_known_vector():
    assert crc32c_u32(b"123456789") == 0xE3069283


def test_fold_reader_bounds_validation_with_mmap(tmp_path):
    file_path = tmp_path / "bounds.fold"
    file_path.write_bytes(b"X" * 64)

    reader = FoldReader(str(file_path), use_mmap=True)
    reader._f = open(file_path, "rb")
    reader._mm = mmap.mmap(reader._f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        with pytest.raises(EOFError, match="além do arquivo"):
            reader._read_at(32, 40)
    finally:
        reader.__exit__(None, None, None)


def test_fold_reader_bounds_validation_negative_values(tmp_path):
    file_path = tmp_path / "bounds-neg.fold"
    file_path.write_bytes(b"X" * 32)

    reader = FoldReader(str(file_path), use_mmap=False)
    reader._f = open(file_path, "rb")
    try:
        with pytest.raises(ValueError, match="offset e length devem ser >= 0"):
            reader._read_at(-1, 4)
        with pytest.raises(ValueError, match="offset e length devem ser >= 0"):
            reader._read_at(0, -4)
    finally:
        reader.__exit__(None, None, None)


def test_fold_reader_index_size_validation(tmp_path):
    file_path = tmp_path / "huge-index.fold"
    fake_header = struct.pack(HEADER_FMT, MAGIC, struct.calcsize(HEADER_FMT), 28, 2_000_000_000)
    file_path.write_bytes(fake_header + (b"\x00" * 128))

    with pytest.raises(ValueError, match="Index muito grande"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_index_offset_validation(tmp_path):
    file_path = tmp_path / "bad-index-offset.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size, header_size - 1, 0)
    file_path.write_bytes(fake_header)

    with pytest.raises(ValueError, match="Index offset inválido"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_header_len_validation(tmp_path):
    file_path = tmp_path / "bad-header-len.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size + 1, header_size, 0)
    file_path.write_bytes(fake_header)

    with pytest.raises(ValueError, match="Header inconsistente"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_reports_magic_values(tmp_path):
    file_path = tmp_path / "wrong-magic.fold"
    bad_magic = b"NOTFOLD!"
    header = struct.pack(HEADER_FMT, bad_magic, struct.calcsize(HEADER_FMT), 24, 0)
    file_path.write_bytes(header)

    with pytest.raises(ValueError, match="Magic esperado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_fold_reader_exit_closes_file_even_if_mmap_close_fails(tmp_path):
    class FailingMM:
        def close(self):
            raise RuntimeError("forced close error")

    file_path = tmp_path / "cleanup.fold"
    file_path.write_bytes(b"abc")

    reader = FoldReader(str(file_path), use_mmap=True)
    reader._f = open(file_path, "rb")
    reader._mm = FailingMM()

    with pytest.raises(RuntimeError, match="forced close error"):
        reader.__exit__(None, None, None)

    assert reader._f is None
    assert reader._mm is None


def test_fold_writer_rejects_oversized_chunk(tmp_path):
    file_path = tmp_path / "too-big.fold"
    with FoldWriter(str(file_path), compress="none") as writer:
        with pytest.raises(ValueError, match="muito grande"):
            writer.add_chunk("big", "JSON", b"x" * (MAX_CHUNK_SIZE + 1))


def test_fold_reader_rejects_invalid_chunk_lengths(tmp_path):
    file_path = tmp_path / "bad-chunk-len.fold"
    neuron = _build_neuron()
    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    with FoldReader(str(file_path), use_mmap=False) as reader:
        chunk = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        offset = chunk["offset"]

    with open(file_path, "r+b") as f:
        f.seek(offset)
        header = f.read(struct.calcsize(">4sIQQII"))
        ctype, flags, uncomp_len, comp_len, crc, ecc_len = struct.unpack(">4sIQQII", header)
        bad_header = struct.pack(
            ">4sIQQII",
            ctype,
            flags,
            uncomp_len,
            MAX_CHUNK_SIZE + 1,
            crc,
            ecc_len,
        )
        f.seek(offset)
        f.write(bad_header)

    with FoldReader(str(file_path), use_mmap=False) as reader:
        with pytest.raises(ValueError, match="tamanho comprimido inválido"):
            reader.read_chunk_bytes("torch_state", verify=False)
