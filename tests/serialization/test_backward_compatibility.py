from pyfolds.serialization.foldio import FoldReader, FoldWriter


def test_reader_without_trust_block_still_works(tmp_path):
    p = tmp_path / "legacy.fold"
    with FoldWriter(str(p), compress="none") as w:
        w.add_chunk("x", "JSON", b"{}")
        w.finalize({"legacy": True})
    with FoldReader(str(p), use_mmap=False) as r:
        assert r.read_chunk_bytes("x") == b"{}"
        assert "trust_block" not in r.index
