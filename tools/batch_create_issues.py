#!/usr/bin/env python3
"""Criação em lote de ISSUEs baseada em JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.create_issue_report import create_issue_template
except ImportError:
    from create_issue_report import create_issue_template

QUEUE_PATH = Path("docs/development/execution_queue.csv")


def load_batch_config(json_file: str) -> dict:
    return json.loads(Path(json_file).read_text(encoding="utf-8"))


def validate_batch_structure(config: dict) -> list[str]:
    errors: list[str] = []
    if "issues" not in config or not isinstance(config["issues"], list):
        errors.append("Campo 'issues' ausente ou inválido")
        return errors
    ids = [issue.get("issue_id", "") for issue in config["issues"]]
    if len(ids) != len(set(ids)):
        errors.append("Há IDs duplicados no batch")
    return errors


def create_batch_issues(issues_list: list[dict]) -> list[Path]:
    created: list[Path] = []
    for issue in issues_list:
        created.append(
            create_issue_template(
                issue_id=issue["issue_id"],
                tema=issue["titulo"],
                prioridade=issue["prioridade"],
                area=issue["area"],
            )
        )
    return created


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="docs/development/batch_issues.json")
    args = parser.parse_args()
    config = load_batch_config(args.config)
    errors = validate_batch_structure(config)
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return 1
    created = create_batch_issues(config["issues"])
    for file in created:
        print(f"✅ {file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
