#!/usr/bin/env python3
"""Utilities to discover and register next ISSUE/ADR identifiers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ISSUE_PATTERN = re.compile(r"ISSUE-(\d{3})")
ADR_PATTERN = re.compile(r"(?:ADR-|\b)(\d{4})(?:\b|[-_])")

ISSUE_SCAN_PATHS = (
    Path("docs/development/prompts/relatorios"),
    Path("docs/development/legado/issues"),
    Path("docs/development/execution_queue.csv"),
)
ADR_SCAN_PATHS = (
    Path("docs/adr"),
    Path("docs/governance/adr"),
    Path("docs/governance/adr/legado"),
)

REGISTRY_PATH = Path("docs/development/legado/id_registry.json")


def _scan_ids(paths: tuple[Path, ...], pattern: re.Pattern[str]) -> list[int]:
    ids: list[int] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            ids.extend(int(match.group(1)) for match in pattern.finditer(text))
            continue

        for child in path.glob("**/*"):
            if child.is_file():
                ids.extend(int(match.group(1)) for match in pattern.finditer(child.name))
    return ids


def next_issue_id() -> str:
    known = _scan_ids(ISSUE_SCAN_PATHS, ISSUE_PATTERN)
    next_value = (max(known) + 1) if known else 1
    return f"ISSUE-{next_value:03d}"


def next_adr_id() -> str:
    known = _scan_ids(ADR_SCAN_PATHS, ADR_PATTERN)
    next_value = (max(known) + 1) if known else 1
    return f"ADR-{next_value:04d}"


def register_ids(issue_id: str, adr_id: str | None = None) -> Path:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_issue_id": issue_id,
        "last_adr_id": adr_id or "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return REGISTRY_PATH
