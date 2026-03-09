from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pyfolds.serialization.foldio import FoldReader, FoldWriter


def run():
    out = {"levels": {}}
    payload = (b"data" * 250_000)
    try:
        import cryptography  # noqa: F401

        has_crypto = True
    except Exception:
        has_crypto = False
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for lvl in ("basic", "standard", "high", "paranoid"):
            p = base / f"{lvl}.fold"
            t0 = time.perf_counter()
            with FoldWriter(str(p), compress="none", security_level=lvl, encrypt=(lvl in {"high"} and has_crypto), encryption_key=(b"k" * 32 if (lvl == "high" and has_crypto) else None), provenance=(lvl=="paranoid"), shard=(lvl=="paranoid")) as w:
                w.add_chunk("blob", "BLOB", payload)
                w.finalize({"bench": True})
            wtime = time.perf_counter() - t0
            t1 = time.perf_counter()
            with FoldReader(str(p), use_mmap=False, decryption_key=(b"k"*32 if (lvl=="high" and has_crypto) else None)) as r:
                r.read_chunk_bytes("blob")
            rtime = time.perf_counter() - t1
            out["levels"][lvl] = {"write_s": wtime, "read_s": rtime, "size_bytes": p.stat().st_size, "encryption_enabled": lvl == "high" and has_crypto}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
