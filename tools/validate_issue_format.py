#!/usr/bin/env python3
"""Validador de formato para ISSUEs de IA."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_SECTIONS = [
    r"^# ISSUE-\d{3}: .+",
    r"^## Metadados$",
    r"^## 1\. Objetivo$",
    r"^## 2\. Escopo$",
    r"^### 2\.1 Inclui:$",
    r"^### 2\.2 Exclui:$",
    r"^## 3\. Artefatos Gerados$",
    r"^## 4\. Riscos$",
    r"^## 5\. Critérios de Aceite$",
    r"^## 6\. PROMPT:EXECUTAR$",
]

PROMPT_YAML_RE = re.compile(r"## 6\. PROMPT:EXECUTAR\s+```yaml\n.*?\n```", re.DOTALL)
FILENAME_RE = re.compile(r"^ISSUE-\d{3}-[a-z0-9-]+\.md$")


def validate_issue_file(path: Path) -> list[str]:
    errors: list[str] = []

    if not FILENAME_RE.match(path.name):
        errors.append(f"Nome de arquivo inválido: {path.name}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        return [f"Erro ao ler arquivo: {exc}"]

    for pattern in REQUIRED_SECTIONS:
        if not re.search(pattern, content, re.MULTILINE):
            errors.append(f"Seção obrigatória não encontrada: {pattern}")

    if not PROMPT_YAML_RE.search(content):
        errors.append("Seção 'PROMPT:EXECUTAR' deve conter bloco ```yaml ... ```")

    return errors


def iter_issue_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("ISSUE-*.md")))
        else:
            files.extend(sorted(Path().glob(item)))
    unique = sorted({f.resolve() for f in files})
    return [Path(p) for p in unique]


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida formato de ISSUEs de IA")
    parser.add_argument("paths", nargs="+", help="Arquivos, diretórios ou globs de ISSUEs")
    args = parser.parse_args()

    files = iter_issue_files(args.paths)
    if not files:
        print("⚠️ Nenhum arquivo ISSUE encontrado")
        return 1

    has_error = False
    for file in files:
        errors = validate_issue_file(file)
        if errors:
            has_error = True
            print(f"❌ {file}")
            for err in errors:
                print(f"   - {err}")
        else:
            print(f"✅ {file}")

    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
