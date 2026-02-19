#!/usr/bin/env python3
"""Valida governança entre execution_queue.csv e HUB_CONTROLE.md."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

CSV_PATH = Path("docs/development/execution_queue.csv")
HUB_PATH = Path("docs/development/HUB_CONTROLE.md")


def parse_csv_ids(text: str) -> list[str]:
    rows = csv.DictReader(text.splitlines())
    return [((row.get("id") or "").strip()) for row in rows if (row.get("id") or "").strip()]


def read_ids(csv_path: Path) -> list[str]:
    return parse_csv_ids(csv_path.read_text(encoding="utf-8"))


def duplicate_ids(ids: list[str]) -> set[str]:
    counts = Counter(ids)
    return {issue_id for issue_id, count in counts.items() if count > 1}


def changed_files(diff_range: str) -> tuple[set[str], bool]:
    result = subprocess.run(
        ["git", "diff", "--name-only", diff_range],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return (set(), False)
    files = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return (files, True)


def validate_hub_sync_change(diff_range: str) -> tuple[bool, str]:
    files, ok = changed_files(diff_range)
    if not ok:
        return (True, "não foi possível calcular diff com a branch base")

    csv_changed = str(CSV_PATH) in files
    hub_changed = str(HUB_PATH) in files

    if csv_changed and not hub_changed:
        return (
            False,
            "execution_queue.csv foi alterado sem atualização concorrente de HUB_CONTROLE.md",
        )
    return (True, "")


def build_diff_range(base_ref: str | None) -> str | None:
    if not base_ref:
        return None
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", base_ref],
        check=False,
        capture_output=True,
        text=True,
    )
    return f"origin/{base_ref}...HEAD"


def base_duplicate_ids(base_ref: str | None) -> set[str] | None:
    if not base_ref:
        return None
    result = subprocess.run(
        ["git", "show", f"origin/{base_ref}:{CSV_PATH.as_posix()}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return duplicate_ids(parse_csv_ids(result.stdout))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida regras de governança da fila/HUB")
    parser.add_argument("--base-ref", default="", help="Branch base para validações de PR")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_ref = args.base_ref.strip() or None

    current_dups = duplicate_ids(read_ids(CSV_PATH))
    legacy_dups = base_duplicate_ids(base_ref)

    if legacy_dups is None:
        new_dups = []
        print("[WARN] Não foi possível carregar baseline da branch base; validação de novos duplicados foi degradada.")
    else:
        new_dups = sorted(current_dups - legacy_dups)

    if new_dups:
        print("[FAIL] Novos IDs duplicados encontrados em execution_queue.csv:")
        for issue_id in new_dups:
            print(f"  - {issue_id}")
        return 1

    if current_dups:
        print("[WARN] Duplicidades legadas detectadas (sem novos casos nesta PR):")
        for issue_id in sorted(current_dups):
            print(f"  - {issue_id}")
    else:
        print("[OK] Nenhuma duplicidade de IDs no execution_queue.csv.")

    diff_range = build_diff_range(base_ref)
    if diff_range:
        ok_sync, sync_msg = validate_hub_sync_change(diff_range)
        if not ok_sync:
            print(f"[FAIL] {sync_msg}")
            return 1
        if sync_msg:
            print(f"[WARN] {sync_msg}")
        else:
            print("[OK] Não há alteração concorrente inválida entre CSV e HUB.")
    else:
        print("[SKIP] Validação de alteração concorrente (base-ref não informado).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
