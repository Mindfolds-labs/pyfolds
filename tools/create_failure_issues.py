#!/usr/bin/env python3
"""Cria issues para falhas sem issue associada e integra com o board pyfolds-board."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_REGISTER_PATH = Path("docs/development/failure_register.csv")
CSV_REQUIRED_FIELDS = {
    "id",
    "tipo",
    "descricao",
    "impacto",
    "status",
    "issue_correcao",
    "assinatura_erro",
    "arquivo_afetado",
    "caminho_log",
    "data_registro",
    "ultima_ocorrencia",
}
SECURITY_CUES = ("security", "seguran", "cve", "vulnerab", "exploit", "critical")


def run_gh(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar gh {' '.join(args)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout.strip()


def load_register(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise SystemExit("CSV de falhas não possui cabeçalho.")
        missing = CSV_REQUIRED_FIELDS - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"CSV sem colunas obrigatórias: {', '.join(sorted(missing))}")
        return list(reader), list(reader.fieldnames)


def save_register(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def infer_label(row: dict[str, str]) -> str:
    blob = " ".join([row.get("tipo", ""), row.get("descricao", ""), row.get("impacto", "")]).lower()
    return "security" if any(cue in blob for cue in SECURITY_CUES) else "bug"


def read_log_excerpt(log_path: str) -> str:
    if not log_path or log_path == "N/A":
        return "Log não informado."
    path = Path(log_path)
    if not path.exists() or not path.is_file():
        return f"Log não encontrado em `{log_path}`."
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    excerpt = "\n".join(lines[-40:]).strip()
    return excerpt or "(arquivo de log vazio)"


def issue_title(row: dict[str, str]) -> str:
    desc = row.get("descricao", "").strip()
    desc = re.sub(r"\s+", " ", desc)
    if len(desc) > 90:
        desc = f"{desc[:87]}..."
    return f"[Failure] {row.get('id', 'N/A')} - {desc or 'Falha sem descrição'}"


def issue_body(row: dict[str, str]) -> str:
    log_excerpt = read_log_excerpt(row.get("caminho_log", ""))
    return f"""## Resumo técnico
- **ID da falha:** {row.get('id', 'N/A')}
- **Tipo:** {row.get('tipo', 'N/A')}
- **Impacto:** {row.get('impacto', 'N/A')}
- **Arquivo afetado:** {row.get('arquivo_afetado', 'N/A')}
- **Assinatura do erro:** `{row.get('assinatura_erro', 'N/A')}`
- **Última ocorrência:** {row.get('ultima_ocorrencia', 'N/A')}

## Reprodução
1. Acessar o log de origem em `{row.get('caminho_log', 'N/A')}`.
2. Reexecutar o fluxo que originou a falha (quando aplicável).
3. Confirmar presença da assinatura `{row.get('assinatura_erro', 'N/A')}` no output.

## Recomendação inicial
- Validar escopo do erro no módulo afetado.
- Definir correção mínima segura com teste de regressão.
- Atualizar `failure_register.csv` ao concluir mitigação.

## Log
```text
{log_excerpt}
```
"""


def parse_issue_identity(issue_ref: str) -> tuple[str, str]:
    url = issue_ref.strip()
    match = re.search(r"/(\d+)$", url)
    if not match:
        raise RuntimeError(f"Não foi possível extrair número da issue a partir de: {url}")
    return match.group(1), url


def detect_repo(explicit_repo: str | None) -> str:
    if explicit_repo:
        return explicit_repo
    repo_name = run_gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    if not repo_name:
        raise RuntimeError("Não foi possível detectar o repositório atual via GitHub CLI.")
    return repo_name


def detect_owner(repo: str, explicit_owner: str | None) -> str:
    if explicit_owner:
        return explicit_owner
    return repo.split("/", 1)[0]


def find_project(owner: str, title: str) -> dict[str, Any]:
    raw = run_gh("project", "list", "--owner", owner, "--format", "json")
    projects = json.loads(raw)
    for project in projects:
        if str(project.get("title", "")).strip().lower() == title.strip().lower():
            return project
    raise RuntimeError(f"Projeto '{title}' não encontrado para owner '{owner}'.")


def add_issue_to_project(owner: str, project_number: int, issue_url: str) -> None:
    run_gh("project", "item-add", str(project_number), "--owner", owner, "--url", issue_url)


def move_issue_to_column(owner: str, project_number: int, project_id: str, issue_url: str, target_column: str) -> None:
    items_raw = run_gh(
        "project",
        "item-list",
        str(project_number),
        "--owner",
        owner,
        "--limit",
        "200",
        "--format",
        "json",
    )
    items = json.loads(items_raw)
    item_id = ""
    for item in items:
        content = item.get("content", {}) or {}
        if content.get("url") == issue_url:
            item_id = item.get("id", "")
            break
    if not item_id:
        raise RuntimeError(f"Item da issue não encontrado no projeto: {issue_url}")

    fields_raw = run_gh("project", "field-list", str(project_number), "--owner", owner, "--format", "json")
    fields = json.loads(fields_raw)

    status_field = None
    option = None
    for field in fields:
        if field.get("name", "").strip().lower() not in {"status", "column"}:
            continue
        options = field.get("options", []) or []
        direct = next((opt for opt in options if opt.get("name", "").strip().lower() == target_column.lower()), None)
        if direct:
            status_field, option = field, direct
            break
        if "/" in target_column:
            aliases = [part.strip().lower() for part in target_column.split("/") if part.strip()]
            alias_opt = next((opt for opt in options if opt.get("name", "").strip().lower() in aliases), None)
            if alias_opt:
                status_field, option = field, alias_opt
                break

    if not status_field or not option:
        raise RuntimeError(f"Coluna '{target_column}' não encontrada em campo de status do projeto.")

    run_gh(
        "project",
        "item-edit",
        "--id",
        item_id,
        "--project-id",
        project_id,
        "--field-id",
        status_field["id"],
        "--single-select-option-id",
        option["id"],
    )


def create_issue(repo: str, row: dict[str, str], label: str) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".md", encoding="utf-8", delete=False) as temp_file:
        temp_file.write(issue_body(row))
        temp_path = temp_file.name

    issue_output = run_gh(
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        issue_title(row),
        "--body-file",
        temp_path,
        "--label",
        label,
    )
    return parse_issue_identity(issue_output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", type=Path, default=DEFAULT_REGISTER_PATH)
    parser.add_argument("--repo", help="Repositório no formato owner/name. Detectado automaticamente se omitido.")
    parser.add_argument("--project-owner", help="Owner do projeto (org/user). Default: owner do repositório.")
    parser.add_argument("--project-title", default="pyfolds-board")
    parser.add_argument("--target-column", default="Backlog/To-Do")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.register.exists():
        raise SystemExit(f"Registro de falhas não encontrado: {args.register}")

    rows, fieldnames = load_register(args.register)
    pending = [row for row in rows if not row.get("issue_correcao", "").strip()]

    if not pending:
        print("Nenhuma falha pendente sem issue associada.")
        return 0

    repo = detect_repo(args.repo)
    owner = detect_owner(repo, args.project_owner)

    project = find_project(owner, args.project_title)
    project_number = int(project["number"])
    project_id = str(project["id"])

    for row in pending:
        label = infer_label(row)
        if args.dry_run:
            print(f"[DRY-RUN] Criaria issue para {row.get('id')} com label '{label}'.")
            continue

        issue_number, issue_url = create_issue(repo, row, label)
        row["issue_correcao"] = f"#{issue_number} ({issue_url})"

        add_issue_to_project(owner, project_number, issue_url)
        move_issue_to_column(owner, project_number, project_id, issue_url, args.target_column)

        print(f"Issue criada para {row.get('id')}: {issue_url} | label={label}")

    if not args.dry_run:
        save_register(args.register, rows, fieldnames)
        print(f"Registro atualizado em {args.register}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
