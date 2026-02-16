import struct

import pytest

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind
from pyfolds.serialization.foldio import HEADER_FMT, MAGIC


def _build_neuron() -> MPJRDNeuron:
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg)


def _save_fold(path):
    save_fold_or_mind(
        _build_neuron(),
        str(path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )


def test_corruption_detection_bit_flip(tmp_path):
    file_path = tmp_path / "bitflip.fold"
    _save_fold(file_path)

    with FoldReader(str(file_path), use_mmap=False) as reader:
        chunk = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        target = chunk["offset"] + chunk["header_len"] + 8

    with open(file_path, "r+b") as f:
        f.seek(target)
        current = f.read(1)
        f.seek(target)
        f.write(bytes([current[0] ^ 0xFF]))

    with pytest.raises(RuntimeError, match="CRC32C inválido"):
        with FoldReader(str(file_path), use_mmap=False) as reader:
            reader.read_chunk_bytes("torch_state", verify=True)


def test_corruption_detection_truncated_file(tmp_path):
    file_path = tmp_path / "truncated.fold"
    _save_fold(file_path)

    raw = file_path.read_bytes()
    file_path.write_bytes(raw[:-16])

    with pytest.raises(ValueError, match="Index truncado|Index JSON corrompido"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_corruption_detection_invalid_magic(tmp_path):
    file_path = tmp_path / "invalid-magic.fold"
    bad_magic = b"NOTFOLD!"
    header = struct.pack(HEADER_FMT, bad_magic, struct.calcsize(HEADER_FMT), 24, 0)
    file_path.write_bytes(header)

    with pytest.raises(ValueError, match="Magic esperado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_corruption_detection_huge_index_len(tmp_path):
    file_path = tmp_path / "huge-index.fold"
    header = struct.pack(HEADER_FMT, MAGIC, struct.calcsize(HEADER_FMT), 28, 2_000_000_000)
    file_path.write_bytes(header + (b"\x00" * 64))

    with pytest.raises(ValueError, match="Index muito grande"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_corruption_detection_partial_read_raises_eoferror(tmp_path):
    file_path = tmp_path / "partial.bin"
    file_path.write_bytes(b"abc")

    reader = FoldReader(str(file_path), use_mmap=False)
    reader._f = open(file_path, "rb")
    try:
        with pytest.raises(EOFError, match="Fim de arquivo inesperado"):
            reader._read_at(0, 8)
    finally:
        reader.__exit__(None, None, None)
