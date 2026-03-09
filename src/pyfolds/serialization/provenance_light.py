from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LightProvenanceEntry:
    action: str
    timestamp: float
    metadata: Dict[str, Any]
    prev_hash: str
    entry_hash: str


class LightProvenance:
    def __init__(self, max_entries: int = 2048):
        self.max_entries = max_entries
        self.entries: List[LightProvenanceEntry] = []

    @staticmethod
    def _hash_entry(action: str, timestamp: float, metadata: Dict[str, Any], prev_hash: str) -> str:
        raw = json.dumps(
            {
                "action": action,
                "timestamp": timestamp,
                "metadata": metadata,
                "prev_hash": prev_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def add(self, action: str, metadata: Optional[Dict[str, Any]] = None) -> LightProvenanceEntry:
        prev = self.entries[-1].entry_hash if self.entries else "GENESIS"
        ts = time.time()
        md = metadata or {}
        h = self._hash_entry(action, ts, md, prev)
        entry = LightProvenanceEntry(action=action, timestamp=ts, metadata=md, prev_hash=prev, entry_hash=h)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        return entry

    def verify(self) -> bool:
        prev = "GENESIS"
        for e in self.entries:
            if e.prev_hash != prev:
                return False
            if self._hash_entry(e.action, e.timestamp, e.metadata, e.prev_hash) != e.entry_hash:
                return False
            prev = e.entry_hash
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_entries": self.max_entries,
            "entries": [
                {
                    "action": e.action,
                    "timestamp": e.timestamp,
                    "metadata": e.metadata,
                    "prev_hash": e.prev_hash,
                    "entry_hash": e.entry_hash,
                }
                for e in self.entries
            ],
        }
