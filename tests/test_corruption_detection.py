import struct

import pytest

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind
from pyfolds.serialization.foldio import HEADER_FMT, MAGIC, MAX_INDEX_SIZE


def _build_neuron() -> MPJRDNeuron:
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg)


def _write_valid_fold(path):
    neuron = _build_neuron()
    save_fold_or_mind(
        neuron,
        str(path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )


def test_detects_bit_flip_corruption(tmp_path):
    file_path = tmp_path / "bitflip.fold"
    _write_valid_fold(file_path)

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


def test_detects_truncated_container(tmp_path):
    file_path = tmp_path / "truncated.fold"
    _write_valid_fold(file_path)

    original = file_path.read_bytes()
    file_path.write_bytes(original[:8])

    with pytest.raises(ValueError, match="inacessível ou truncado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_detects_invalid_magic(tmp_path):
    file_path = tmp_path / "bad-magic.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, b"BADC", header_size, header_size, 0)
    file_path.write_bytes(fake_header)

    with pytest.raises(ValueError, match="Magic esperado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_rejects_index_len_dos_payload(tmp_path):
    file_path = tmp_path / "dos-index.fold"
    header_size = struct.calcsize(HEADER_FMT)
    index_len = MAX_INDEX_SIZE + 1
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size, header_size, index_len)
    file_path.write_bytes(fake_header + (b"\x00" * 64))

    with pytest.raises(ValueError, match="Index muito grande"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_partial_read_raises_eoferror(tmp_path):
    class PartialReadFile:
        def __init__(self, raw):
            self._raw = raw

        def seek(self, offset):
            self._raw.seek(offset)

        def read(self, length):
            requested = max(length, 1)
            return self._raw.read(max(1, requested // 2))

    file_path = tmp_path / "partial-read.fold"
    file_path.write_bytes(b"A" * 32)

    reader = FoldReader(str(file_path), use_mmap=False)
    with open(file_path, "rb") as raw_file:
        reader._f = PartialReadFile(raw_file)
        with pytest.raises(EOFError, match="Fim de arquivo inesperado"):
            reader._read_at(0, 16)
