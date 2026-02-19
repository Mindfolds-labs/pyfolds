#!/usr/bin/env python3
"""Validação de relatórios de ISSUE com foco em ABNT/IEEE e links internos."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:
    yaml = None

REQUIRED_SECTIONS = [
    "## Objetivo",
    "## Contexto Técnico",
    "## Análise Técnica",
    "## Requisitos Funcionais",
    "## Requisitos Não-Funcionais",
    "## Artefatos Esperados",
    "## Critérios de Aceite",
    "## Riscos e Mitigações",
    "## PROMPT:EXECUTAR",
    "## Rastreabilidade (IEEE 830)",
]


def validate_yaml_header(filepath: Path) -> list[str]:
    content = filepath.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        return ["Frontmatter YAML ausente"]
    end = content.find("\n---\n", 4)
    if end == -1:
        return ["Frontmatter YAML não encerrado"]
    block = content[4:end]
    if yaml is None:
        data = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
    else:
        try:
            data = yaml.safe_load(block)
        except yaml.YAMLError as exc:
            return [f"YAML inválido: {exc}"]
    if not isinstance(data, dict) or "id" not in data:
        return ["YAML sem campo obrigatório 'id'"]
    return []


def check_required_sections(filepath: Path) -> list[str]:
    content = filepath.read_text(encoding="utf-8")
    return [f"Seção obrigatória ausente: {s}" for s in REQUIRED_SECTIONS if s not in content]


def validate_structure(filepath: Path) -> list[str]:
    errors = []
    if not re.match(r"ISSUE-\d{3}-[a-z0-9-]+\.md$", filepath.name):
        errors.append(f"Nome inválido: {filepath.name}")
    content = filepath.read_text(encoding="utf-8")
    if "```yaml" not in content:
        errors.append("Bloco yaml do PROMPT:EXECUTAR ausente")
    return errors


def validate_links(filepath: Path) -> list[str]:
    content = filepath.read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]*\]\(([^)]+)\)", content)
    errors: list[str] = []
    for link in links:
        if link.startswith(("http://", "https://", "mailto:")):
            continue
        target = (filepath.parent / link).resolve()
        if not target.exists():
            errors.append(f"Link interno inexistente: {link}")
    return errors


def iter_issue_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("ISSUE-*.md")))
        else:
            files.extend(sorted(Path().glob(item)))
    return sorted({f.resolve() for f in files})


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida formato de ISSUEs de IA")
    parser.add_argument("paths", nargs="+", help="Arquivos, diretórios ou globs")
    args = parser.parse_args()
    files = iter_issue_files(args.paths)
    if not files:
        print("⚠️ Nenhum arquivo ISSUE encontrado")
        return 1

    failed = False
    for f in files:
        errors = [
            *validate_yaml_header(Path(f)),
            *validate_structure(Path(f)),
            *check_required_sections(Path(f)),
            *validate_links(Path(f)),
        ]
        if errors:
            failed = True
            print(f"❌ {f}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"✅ {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
