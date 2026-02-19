#!/usr/bin/env python3
"""Synchronize docs/development/HUB_CONTROLE.md blocks from CSV files.

Flow:
1) Read docs/development/execution_queue.csv as source of truth for active queue.
2) Read docs/development/failure_register.csv as source of truth for detected failures.
3) Render deterministic Markdown (summary table, detail cards, failures table).
4) Replace only content between HUB markers in HUB_CONTROLE.md.
5) In --check mode, do not write; exit 1 if file is out of sync.

Recommended routine (local/CI pre-commit):
- Local: ``python tools/sync_hub.py && python tools/sync_hub.py --check``
- CI: run ``python tools/sync_hub.py --check`` to fail fast when HUB is stale.

The generation is deterministic to avoid noisy diffs:
- queue rows are sorted by priority/date/id;
- failure rows are sorted by criticality/date/id.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("docs/development/execution_queue.csv")
FAILURE_CSV_PATH = Path("docs/development/failure_register.csv")
HUB_PATH = Path("docs/development/HUB_CONTROLE.md")

QUEUE_BEGIN_MARKER = "<!-- HUB:QUEUE:BEGIN -->"
QUEUE_END_MARKER = "<!-- HUB:QUEUE:END -->"
CARDS_BEGIN_MARKER = "<!-- HUB:CARDS:BEGIN -->"
CARDS_END_MARKER = "<!-- HUB:CARDS:END -->"
FAILURES_BEGIN_MARKER = "<!-- HUB:FAILURES:BEGIN -->"
FAILURES_END_MARKER = "<!-- HUB:FAILURES:END -->"

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

REQUIRED_FAILURE_COLUMNS = [
    "id",
    "tipo",
    "descricao",
    "impacto",
    "status",
    "issue_correcao",
    "data_registro",
    "ultima_ocorrencia",
]

PRIORITY_ORDER = {
    "alta": 0,
    "high": 0,
    "media": 1,
    "média": 1,
    "medium": 1,
    "baixa": 2,
    "low": 2,
}

CRITICALITY_ORDER = {
    "critica": 0,
    "crítica": 0,
    "critical": 0,
    "alta": 1,
    "high": 1,
    "media": 2,
    "média": 2,
    "medium": 2,
    "baixa": 3,
    "low": 3,
}

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


def _date_sort_key(value: str) -> tuple[int, str]:
    normalized = _normalize_date(value)
    if normalized:
        return (0, normalized)
    return (1, "9999-99-99")


def _priority_rank(value: str) -> int:
    normalized = _norm(value).lower()
    for key, rank in PRIORITY_ORDER.items():
        if key in normalized:
            return rank
    return 99


def _criticality_rank(value: str) -> int:
    normalized = _norm(value).lower()
    for key, rank in CRITICALITY_ORDER.items():
        if key in normalized:
            return rank
    return 99


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


def _status_theme(status: str) -> dict[str, str]:
    normalized = _norm(status).lower()
    for key, theme in STATUS_THEME.items():
        if key in normalized:
            return theme
    return DEFAULT_THEME


def _resolve_required_artifact(base_dir: Path, prefix: str) -> str:
    """Resolve mandatory artifact path for active queue entries.

    Legacy files are ignored because lookup is restricted to active directories
    (`prompts/relatorios` and `prompts/execucoes`).
    """
    matches = sorted(base_dir.glob(f"{prefix}-*.md"))
    if not matches:
        raise RuntimeError(
            f"Artefato obrigatório ausente para {prefix} em {base_dir.as_posix()}"
        )
    if len(matches) > 1:
        joined = ", ".join(path.name for path in matches)
        raise RuntimeError(f"Artefato ambíguo para {prefix}: {joined}")
    return f"./{matches[0].relative_to(Path('docs/development')).as_posix()}"


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

    seen_ids: set[str] = set()
    for row in rows:
        issue_id = _norm(row.get("id", ""))
        if not issue_id:
            raise RuntimeError("Linha da fila sem ID")
        if issue_id in seen_ids:
            raise RuntimeError(f"ID duplicado na fila ativa: {issue_id}")
        seen_ids.add(issue_id)

    return rows


def read_failure_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise RuntimeError(f"CSV de falhas não encontrado: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_FAILURE_COLUMNS if name not in fieldnames]
        if missing:
            raise RuntimeError(
                "failure_register.csv sem colunas obrigatórias: "
                + ", ".join(missing)
                + f". Esperado ao menos: {', '.join(REQUIRED_FAILURE_COLUMNS)}"
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = {key: _norm(row.get(key, "")) for key in fieldnames}
            if not any(normalized.values()):
                continue
            rows.append(normalized)

    return rows


def sort_queue_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _priority_rank(row.get("prioridade", "")),
            _date_sort_key(row.get("data", "")),
            _norm(row.get("id", "")).lower(),
        ),
    )


def sort_failure_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            _criticality_rank(row.get("impacto", "")),
            _date_sort_key(row.get("ultima_ocorrencia", "") or row.get("data_registro", "")),
            _norm(row.get("id", "")).lower(),
        ),
    )


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
        report_path = _resolve_required_artifact(REPORTS_DIR, issue_id)
        exec_path = _resolve_required_artifact(EXECS_DIR, exec_prefix)

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


def build_failures_table(rows: list[dict[str, str]]) -> str:
    headers = [
        "ID",
        "Tipo",
        "Descrição",
        "Impacto",
        "Status",
        "Issue de Correção",
        "Data",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":--"] * len(headers)) + " |",
    ]

    if not rows:
        lines.append("| - | - | - | - | - | - | - |")
        return "\n".join(lines)

    for row in rows:
        last_seen = _normalize_date(row.get("ultima_ocorrencia", ""))
        registered = _normalize_date(row.get("data_registro", ""))
        effective_date = last_seen or registered or "-"
        values = [
            _format_cell("id", row.get("id", "")),
            _format_cell("tipo", row.get("tipo", "")),
            _format_cell("descricao", row.get("descricao", "")),
            _format_cell("impacto", row.get("impacto", "")),
            _format_cell("status", row.get("status", "")),
            _format_cell("issue_correcao", row.get("issue_correcao", "")),
            _format_cell("data", effective_date),
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


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
        f"{FAILURES_BEGIN_MARKER}\nold\n{FAILURES_END_MARKER}\n"
        "suffix\n"
    )
    table = "| A |\n| :-- |\n| B |"
    cards = "> card"
    failures = "| F |"
    once = replace_block(sample, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
    once = replace_block(once, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)
    once = replace_block(once, FAILURES_BEGIN_MARKER, FAILURES_END_MARKER, failures)
    twice = replace_block(once, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
    twice = replace_block(twice, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)
    twice = replace_block(twice, FAILURES_BEGIN_MARKER, FAILURES_END_MARKER, failures)
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
        rows = sort_queue_rows(read_rows(CSV_PATH))
        failures_rows = sort_failure_rows(read_failure_rows(FAILURE_CSV_PATH))
        table = build_table(rows)
        cards = build_cards(rows)
        failures_table = build_failures_table(failures_rows)

        original = HUB_PATH.read_text(encoding="utf-8")
        updated = replace_block(original, QUEUE_BEGIN_MARKER, QUEUE_END_MARKER, table)
        updated = replace_block(updated, CARDS_BEGIN_MARKER, CARDS_END_MARKER, cards)
        updated = replace_block(updated, FAILURES_BEGIN_MARKER, FAILURES_END_MARKER, failures_table)

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
