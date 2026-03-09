import time

from pyfolds.serialization.foldio import FoldReader, FoldWriter


def _roundtrip(tmp_path, level):
    p = tmp_path / f"{level}.fold"
    t0 = time.perf_counter()
    with FoldWriter(str(p), compress="none", security_level=level) as w:
        w.add_chunk("x", "JSON", b"{" + b"a" * 10000 + b"}")
        w.finalize({"lvl": level})
    wt = time.perf_counter() - t0
    t1 = time.perf_counter()
    with FoldReader(str(p), use_mmap=False) as r:
        r.read_chunk_bytes("x")
    rt = time.perf_counter() - t1
    return wt, rt


def test_performance_smoke(tmp_path):
    for lvl in ("basic", "standard", "high", "paranoid"):
        w, r = _roundtrip(tmp_path, lvl)
        assert w >= 0
        assert r >= 0
