import struct

import pytest

from pyfolds.serialization import FoldReader, save_fold_or_mind
from pyfolds.serialization.foldio import HEADER_FMT, MAGIC
from tests.unit.serialization.test_foldio import _build_neuron


def test_detects_bit_flip_corruption(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "bitflip.fold"

    save_fold_or_mind(neuron, str(file_path), compress="none", include_history=False, include_telemetry=False)

    with FoldReader(str(file_path), use_mmap=False) as reader:
        torch_state = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        byte_to_flip = torch_state["offset"] + torch_state["header_len"] + 16

    with open(file_path, "r+b") as f:
        f.seek(byte_to_flip)
        original = f.read(1)
        f.seek(byte_to_flip)
        f.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(RuntimeError, match="CRC32C inválido"):
        with FoldReader(str(file_path), use_mmap=False) as reader:
            reader.read_chunk_bytes("torch_state", verify=True)


def test_detects_truncated_file(tmp_path):
    neuron = _build_neuron()
    file_path = tmp_path / "truncated.fold"

    save_fold_or_mind(neuron, str(file_path), compress="none", include_history=False, include_telemetry=False)

    raw = file_path.read_bytes()
    file_path.write_bytes(raw[:-64])

    with pytest.raises((ValueError, EOFError), match="truncado|Fim de arquivo|Index truncado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_detects_invalid_magic(tmp_path):
    file_path = tmp_path / "bad-magic.fold"
    header_size = struct.calcsize(HEADER_FMT)
    bad_header = struct.pack(HEADER_FMT, b"NOTMAGIC", header_size, header_size, 0)
    file_path.write_bytes(bad_header)

    with pytest.raises(ValueError, match="Magic esperado"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_detects_huge_index_len_dos_guard(tmp_path):
    file_path = tmp_path / "huge-index.fold"
    header_size = struct.calcsize(HEADER_FMT)
    fake_header = struct.pack(HEADER_FMT, MAGIC, header_size, header_size, 2_000_000_000)
    file_path.write_bytes(fake_header + (b"\x00" * 32))

    with pytest.raises(ValueError, match="Index muito grande"):
        with FoldReader(str(file_path), use_mmap=False):
            pass


def test_partial_read_raises_eoferror(tmp_path):
    file_path = tmp_path / "partial.bin"
    file_path.write_bytes(b"abc")

    reader = FoldReader(str(file_path), use_mmap=False)
    reader._f = open(file_path, "rb")
    try:
        with pytest.raises(EOFError, match="Fim de arquivo inesperado"):
            reader._read_at(0, 10)
    finally:
        reader.__exit__(None, None, None)
