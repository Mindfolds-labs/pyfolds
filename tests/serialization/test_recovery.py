from pyfolds.serialization.foldio import FoldWriter
from pyfolds.serialization.recovery import (
    attempt_chunk_repair,
    generate_recovery_report,
    locate_corrupted_chunks,
    scan_fold_integrity,
)


def test_recovery_report_and_scan(tmp_path):
    p = tmp_path / "r.fold"
    with FoldWriter(str(p), compress="none") as w:
        w.add_chunk("x", "JSON", b"{}")
        w.finalize({"a": 1})
    rep = scan_fold_integrity(str(p))
    assert rep["ok"]
    assert locate_corrupted_chunks(str(p)) == []
    assert "chunks" in generate_recovery_report(str(p))
    out = attempt_chunk_repair(str(p), str(tmp_path / "copy.fold"))
    assert out["repaired"] is False
