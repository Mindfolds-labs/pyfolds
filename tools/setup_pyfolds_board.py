#!/usr/bin/env python3
"""Bootstrap do projeto GitHub `pyfolds-board` com validações de colunas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

REQUIRED_COLUMNS = ["Backlog/To-Do", "In Progress", "Review/Blocked", "Done"]


def run_gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar gh {' '.join(args)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout.strip()


def parse_owner_from_repo(repo: str) -> str:
    return repo.split("/", 1)[0]


def detect_repo() -> str:
    return run_gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")


def find_project(owner: str, title: str) -> dict[str, Any] | None:
    projects_raw = run_gh("project", "list", "--owner", owner, "--format", "json")
    projects: list[dict[str, Any]] = json.loads(projects_raw)
    for project in projects:
        if str(project.get("title", "")).strip().lower() == title.strip().lower():
            return project
    return None


def ensure_project(owner: str, title: str) -> dict[str, Any]:
    project = find_project(owner, title)
    if project:
        print(f"Projeto '{title}' já existe (#{project['number']}).")
        return project

    print("Criando projeto (base para template Automated Kanban)...")
    created_raw = run_gh("project", "create", "--owner", owner, "--title", title, "--format", "json")
    created: dict[str, Any] = json.loads(created_raw)
    print("Projeto criado. Configure no UI o template 'Automated Kanban'.")
    return created


def get_fields(owner: str, number: int) -> list[dict[str, Any]]:
    fields_raw = run_gh("project", "field-list", str(number), "--owner", owner, "--format", "json")
    return json.loads(fields_raw)


def validate_columns(owner: str, number: int) -> None:
    fields = get_fields(owner, number)
    status_field = next((f for f in fields if f.get("name", "").strip().lower() == "status"), None)
    if not status_field:
        raise RuntimeError("Campo 'Status' não encontrado no projeto.")

    options = [str(opt.get("name", "")).strip() for opt in status_field.get("options", [])]
    missing = [col for col in REQUIRED_COLUMNS if col not in options]
    if missing:
        raise RuntimeError(
            "Colunas obrigatórias ausentes no campo Status: " + ", ".join(missing)
        )

    print("Colunas validadas com sucesso:")
    for col in REQUIRED_COLUMNS:
        print(f"- {col}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="Repositório owner/name para detectar owner automaticamente.")
    parser.add_argument("--owner", help="Owner do projeto (org/user). Sobrescreve --repo.")
    parser.add_argument("--title", default="pyfolds-board")
    args = parser.parse_args()

    repo = args.repo or detect_repo()
    owner = args.owner or parse_owner_from_repo(repo)

    project = ensure_project(owner, args.title)
    number = int(project["number"])

    print(
        "\nIMPORTANTE: se o projeto foi recém-criado, aplique o template 'Automated Kanban' na interface web antes da validação final."
    )
    validate_columns(owner, number)
    print("\nProjeto pronto para automações de status por issue/PR.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
