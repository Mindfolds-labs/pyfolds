#!/usr/bin/env python3
"""Synchronize docs/development/HUB_CONTROLE.md blocks from CSV.

Flow:
1) Read docs/development/execution_queue.csv as source of truth.
2) Render deterministic Markdown table (resumo) and cards (detalhamento).
3) Replace only content between HUB markers in HUB_CONTROLE.md.
4) In --check mode, do not write; exit 1 if file is out of sync.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("docs/development/execution_queue.csv")
HUB_PATH = Path("docs/development/HUB_CONTROLE.md")

QUEUE_BEGIN_MARKER = "<!-- HUB:QUEUE:BEGIN -->"
QUEUE_END_MARKER = "<!-- HUB:QUEUE:END -->"
CARDS_BEGIN_MARKER = "<!-- HUB:CARDS:BEGIN -->"
CARDS_END_MARKER = "<!-- HUB:CARDS:END -->"

REPORTS_DIR = Path("docs/development/prompts/relatorios")
EXECS_DIR = Path("docs/development/prompts/execucoes")

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

STATUS_THEME = {
    "concluida": {"badge": "✅ Concluída", "callout": "[!TIP]", "tone": "Sucesso"},
    "concluída": {"badge": "✅ Concluída", "callout": "[!TIP]", "tone": "Sucesso"},
    "done": {"badge": "✅ Concluída", "callout": "[!TIP]", "tone": "Sucesso"},
    "em progresso": {"badge": "🚧 Em Progresso", "callout": "[!WARNING]", "tone": "Andamento"},
    "progresso": {"badge": "🚧 Em Progresso", "callout": "[!WARNING]", "tone": "Andamento"},
    "planejada": {"badge": "⏳ Planejada", "callout": "[!NOTE]", "tone": "Planejamento"},
    "bloqueada": {"badge": "❌ Bloqueada", "callout": "[!CAUTION]", "tone": "Bloqueio"},
    "cancelada": {"badge": "⚪ Cancelada", "callout": "[!IMPORTANT]", "tone": "Cancelada"},
}
DEFAULT_THEME = {"badge": "🔹 Status não categorizado", "callout": "[!NOTE]", "tone": "Informativo"}


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


def _slugify(value: str) -> str:
    text = value.lower().strip()
    swap = {
        "á": "a",
        "à": "a",
        "â": "a",
        "ã": "a",
        "é": "e",
        "ê": "e",
        "í": "i",
        "ó": "o",
        "ô": "o",
        "õ": "o",
        "ú": "u",
        "ç": "c",
    }
    for k, v in swap.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "sem-slug"


def _status_theme(status: str) -> dict[str, str]:
    normalized = _norm(status).lower()
    for key, theme in STATUS_THEME.items():
        if key in normalized:
            return theme
    return DEFAULT_THEME


def _discover_artifact_path(base_dir: Path, prefix: str, fallback_slug: str) -> str:
    preferred = base_dir / f"{prefix}-{fallback_slug}.md"
    if preferred.exists():
        return f"./{preferred.relative_to(Path('docs/development')).as_posix()}"

    matches = sorted(base_dir.glob(f"{prefix}-*.md"))
    if matches:
        return f"./{matches[0].relative_to(Path('docs/development')).as_posix()}"

    return f"./{base_dir.relative_to(Path('docs/development')).as_posix()}/{prefix}-{fallback_slug}.md"


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


def build_cards(rows: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for row in rows:
        issue_id = _norm(row.get("id")) or "ISSUE-SEM-ID"
        tema = _norm(row.get("tema")) or "Tema não informado"
        status = _norm(row.get("status"))
        responsavel = _norm(row.get("responsavel")) or "-"
        data = _normalize_date(_norm(row.get("data"))) or "-"
        prioridade = _norm(row.get("prioridade")) or "-"
        area = _norm(row.get("area")) or "-"

        issue_num = issue_id.replace("ISSUE", "").lstrip("-")
        exec_prefix = f"EXEC-{issue_num}" if issue_num else "EXEC"
        tema_slug = _slugify(tema)

        report_path = _discover_artifact_path(REPORTS_DIR, issue_id, tema_slug)
        exec_path = _discover_artifact_path(EXECS_DIR, exec_prefix, tema_slug)

        theme = _status_theme(status)
        badge = theme["badge"]
        callout = theme["callout"]

        blocks.extend(
            [
                f"> {callout}",
                f"> **{issue_id}** · {tema}",
                ">",
                f"> **Status:** {badge}  ",
                f"> **Responsável:** {responsavel}  ",
                f"> **Data:** {data}  ",
                f"> **Prioridade:** `{prioridade}` · **Área:** `{area}`  ",
                ">",
                f"> 📄 [Relatório]({report_path}) · 🛠️ [Execução]({exec_path})",
                "",
            ]
        )

    return "\n".join(blocks).rstrip() + "\n"


def replace_block(content: str, begin_marker: str, end_marker: str, generated: str) -> str:
    if begin_marker not in content or end_marker not in content:
        raise RuntimeError(f"Marcadores {begin_marker}/{end_marker} não encontrados")

    begin_count = content.count(begin_marker)
    end_count = content.count(end_marker)
    if begin_count != 1 or end_count != 1:
        raise RuntimeError(f"Marcadores {begin_marker}/{end_marker} devem existir exatamente uma vez")

    start = content.index(begin_marker) + len(begin_marker)
    end = content.index(end_marker)
    if start > end:
        raise RuntimeError(f"Ordem inválida dos marcadores {begin_marker}/{end_marker}")

    replacement = f"\n{generated}\n"
    return content[:start] + replacement + content[end:]


def self_check() -> None:
    sample = (
        "prefix\n"
        f"{QUEUE_BEGIN_MARKER}\nold\n{QUEUE_END_MARKER}\n"
        f"{CARDS_BEGIN_MARKER}\nold\n{CARDS_END_MARKER}\n"
        "suffix\n"
    )
    table = "| A |\n| :-- |\n| B |"
    cards = "> card"
    once = replace_block(sample, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
    once = replace_block(once, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)
    twice = replace_block(once, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
    twice = replace_block(twice, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)
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
        cards = build_cards(rows)

        original = HUB_PATH.read_text(encoding="utf-8")
        updated = replace_block(original, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
        updated = replace_block(updated, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)

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
