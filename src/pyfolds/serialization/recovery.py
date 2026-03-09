from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .foldio import FoldReader
from .sharding_raid import RAIDSharding
from .trust_block import TrustBlock


def scan_fold_integrity(path: str, verify: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": path, "chunks": [], "ok": True}
    with FoldReader(path, use_mmap=False) as reader:
        for name in reader.list_chunks():
            try:
                reader.read_chunk_bytes(name, verify=verify)
                report["chunks"].append({"name": name, "ok": True})
            except Exception as exc:
                report["ok"] = False
                report["chunks"].append({"name": name, "ok": False, "error": str(exc)})
    return report


def locate_corrupted_chunks(path: str) -> List[str]:
    rep = scan_fold_integrity(path, verify=True)
    return [c["name"] for c in rep["chunks"] if not c["ok"]]


def attempt_chunk_repair(path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    corrupted = locate_corrupted_chunks(path)
    result = {"source": path, "corrupted": corrupted, "repaired": False}
    if not corrupted:
        return result
    if output_path is None:
        return result
    Path(output_path).write_bytes(Path(path).read_bytes())
    result["repaired"] = False
    result["output_path"] = output_path
    result["note"] = "Cópia forense gerada; reparo automático não seguro sem redundância adicional."
    return result


def reconstruct_from_shards(available_shards: Sequence[bytes], available_indices: Sequence[int], data_shards: int = 4, parity_shards: int = 1) -> bytes:
    return RAIDSharding(data_shards=data_shards, parity_shards=parity_shards).reconstruct(available_shards, available_indices)


def verify_trust_chain(trust_block_bytes: bytes, public_key_pem: str) -> bool:
    tb = TrustBlock.from_bytes(trust_block_bytes)
    return tb.verify(public_key_pem)


def generate_recovery_report(path: str) -> str:
    return json.dumps(scan_fold_integrity(path, verify=True), indent=2, ensure_ascii=False)
