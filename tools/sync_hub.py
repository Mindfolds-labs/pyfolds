#!/usr/bin/env python3
"""Synchronize docs/development/HUB_CONTROLE.md queue block from CSV.

Flow:
1) Read docs/development/execution_queue.csv as source of truth.
2) Render deterministic Markdown table from CSV row order.
3) Replace only content between HUB queue markers in HUB_CONTROLE.md.
4) In --check mode, do not write; exit 1 if file is out of sync.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("docs/development/execution_queue.csv")
HUB_PATH = Path("docs/development/HUB_CONTROLE.md")
BEGIN_MARKER = "<!-- HUB:QUEUE:BEGIN -->"
END_MARKER = "<!-- HUB:QUEUE:END -->"

REQUIRED_COLUMNS = [
    "id",
    "tema",
    "status",
    "responsavel",
    "data",
    "artefatos",
    "github_issue",
    "pr",
    "prioridade",
    "area",
]


def _norm(value: str | None) -> str:
    raw = "" if value is None else str(value)
    cleaned = raw.strip()
    if cleaned in {"-", "—", "N/A", "n/a"}:
        return ""
    return cleaned


def _escape_md(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _normalize_date(value: str) -> str:
    cleaned = _norm(value)
    if not cleaned:
        return ""

    formats = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")
    for fmt in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return cleaned


def _format_cell(key: str, value: str) -> str:
    cleaned = _norm(value)
    if not cleaned:
        return "-"

    if key == "data":
        return _escape_md(_normalize_date(cleaned))

    if key == "artefatos":
        parts = [part.strip() for part in cleaned.split(";") if part.strip()]
        if parts:
            return "<br>".join(_escape_md(part) for part in parts)

    return _escape_md(cleaned)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise RuntimeError(f"CSV não encontrado: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise RuntimeError(
                "CSV sem colunas obrigatórias: "
                + ", ".join(missing)
                + f". Esperado ao menos: {', '.join(REQUIRED_COLUMNS)}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {key: _norm(row.get(key, "")) for key in REQUIRED_COLUMNS}
            if not any(normalized.values()):
                continue
            rows.append(normalized)

    return rows


def build_table(rows: list[dict[str, str]]) -> str:
    headers = ["ID", "Status", "Tema", "Responsável", "Data"]
    keys = ["id", "status", "tema", "responsavel", "data"]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":--"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [_format_cell(key, row.get(key, "")) for key in keys]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def replace_queue_block(content: str, generated_table: str) -> str:
    if BEGIN_MARKER not in content or END_MARKER not in content:
        raise RuntimeError(
            "Marcadores HUB:QUEUE não encontrados em docs/development/HUB_CONTROLE.md"
        )

    begin_count = content.count(BEGIN_MARKER)
    end_count = content.count(END_MARKER)
    if begin_count != 1 or end_count != 1:
        raise RuntimeError("Marcadores HUB:QUEUE devem existir exatamente uma vez")

    start = content.index(BEGIN_MARKER) + len(BEGIN_MARKER)
    end = content.index(END_MARKER)
    if start > end:
        raise RuntimeError("Ordem inválida dos marcadores HUB:QUEUE")

    replacement = f"\n{generated_table}\n"
    return content[:start] + replacement + content[end:]


def self_check() -> None:
    sample = (
        "prefix\n"
        f"{BEGIN_MARKER}\nold\n{END_MARKER}\n"
        "suffix\n"
    )
    table = "| A |\n| :-- |\n| B |"
    once = replace_queue_block(sample, table)
    twice = replace_queue_block(once, table)
    if once != twice:
        raise RuntimeError("Self-check falhou: substituição não é idempotente")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sincroniza HUB_CONTROLE.md a partir do CSV")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Não grava arquivo; retorna 1 se HUB estiver desatualizado",
    )
    args = parser.parse_args()

    self_check()

    try:
        rows = read_rows(CSV_PATH)
        table = build_table(rows)

        original = HUB_PATH.read_text(encoding="utf-8")
        updated = replace_queue_block(original, table)

        if args.check:
            return 1 if updated != original else 0

        if updated != original:
            HUB_PATH.write_text(updated, encoding="utf-8", newline="")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
