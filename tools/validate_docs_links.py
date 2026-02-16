#!/usr/bin/env python3
"""Validate local markdown links under docs/.

Checks only local relative links (no http(s), mailto, anchors-only).
Exits with non-zero status when any target path is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS_ROOT = Path("docs")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _is_ignored(target: str) -> bool:
    t = target.strip()
    return (
        not t
        or t.startswith("#")
        or t.startswith("http://")
        or t.startswith("https://")
        or t.startswith("mailto:")
        or t.startswith("data:")
    )


def _normalize_target(raw: str) -> str:
    target = raw.strip().split()[0]
    target = target.split("#", 1)[0]
    return target


def main() -> int:
    if not DOCS_ROOT.exists():
        print("docs/ directory not found", file=sys.stderr)
        return 2

    missing: list[tuple[Path, str]] = []

    for md_file in DOCS_ROOT.rglob("*.md"):
        # ADR legacy filenames are intentionally preserved and may use historical link aliases.
        if "docs/governance/adr" in str(md_file).replace("\\", "/"):
            continue
        content = md_file.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(content):
            target = _normalize_target(raw_target)
            if _is_ignored(target):
                continue

            target_path = (md_file.parent / target).resolve()
            if not target_path.exists():
                missing.append((md_file, target))

    if missing:
        print("Broken local links found:")
        for source, target in missing:
            print(f"- {source}: {target}")
        return 1

    print("All local docs links look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
