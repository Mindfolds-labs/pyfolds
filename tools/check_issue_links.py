#!/usr/bin/env python3
"""Verifica referências ISSUE-XXX entre relatórios."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ISSUE_REF_RE = re.compile(r"\bISSUE-(\d{3})\b")
ISSUE_FILE_RE = re.compile(r"^ISSUE-(\d{3})-[a-z0-9-]+\.md$")


def collect_issue_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in sorted(root.glob("ISSUE-*.md")):
        m = ISSUE_FILE_RE.match(p.name)
        if m:
            files[m.group(1)] = p
    return files


def check_references(root: Path) -> tuple[bool, list[str]]:
    issues = collect_issue_files(root)
    errors: list[str] = []

    for issue_id, file_path in issues.items():
        text = file_path.read_text(encoding="utf-8")
        refs = sorted(set(ISSUE_REF_RE.findall(text)))
        for ref in refs:
            if ref == issue_id:
                continue
            if ref not in issues:
                errors.append(
                    f"{file_path}: referência para ISSUE-{ref} não encontrada em {root}"
                )

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica links/referências entre ISSUEs")
    parser.add_argument(
        "path",
        nargs="?",
        default="docs/development/prompts/relatorios",
        help="Diretório contendo ISSUE-*.md",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists() or not root.is_dir():
        print(f"❌ Diretório inválido: {root}")
        return 1

    ok, errors = check_references(root)
    if not ok:
        print("❌ Referências inválidas encontradas:")
        for err in errors:
            print(f"   - {err}")
        return 1

    print(f"✅ Referências entre ISSUEs válidas em {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
