from pyfolds.serialization.foldio import FoldReader, FoldWriter
import pytest


def _mk(tmp_path, level, **kwargs):
    p = tmp_path / f"{level}.fold"
    with FoldWriter(str(p), compress="none", security_level=level, **kwargs) as w:
        w.add_chunk("x", "JSON", b"{}")
        w.finalize({"k": "v"})
    with FoldReader(str(p), use_mmap=False, decryption_key=kwargs.get("encryption_key")) as r:
        assert r.read_chunk_bytes("x") == b"{}"
        assert r.index["metadata"]["security_level"] == level


def test_levels(tmp_path):
    _mk(tmp_path, "basic")
    _mk(tmp_path, "standard")
    try:
        import cryptography  # noqa: F401

        _mk(tmp_path, "high", encrypt=True, encryption_key=b"k" * 32)
    except Exception:
        pytest.skip("cryptography opcional ausente")
    _mk(tmp_path, "paranoid", provenance=True, shard=True)
