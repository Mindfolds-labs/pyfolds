#!/usr/bin/env python3
"""Criação de relatórios de ISSUE com frontmatter YAML e checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RELATORIOS_DIR = Path("docs/development/prompts/relatorios")
LOGS_DIR = Path("docs/development/prompts/logs")


@dataclass(slots=True)
class IssueData:
    issue_id: str
    tema: str
    prioridade: str
    area: str
    responsavel: str = "Codex"


def slugify(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return clean or "sem-tema"


def validate_metadata(config: IssueData) -> None:
    if not re.fullmatch(r"ISSUE-\d{3}", config.issue_id):
        raise ValueError("issue_id deve seguir o padrão ISSUE-XXX")
    if not config.tema.strip() or not config.prioridade.strip() or not config.area.strip():
        raise ValueError("tema, prioridade e area são obrigatórios")


def generate_yaml_frontmatter(issue_data: IssueData) -> str:
    payload = {
        "id": issue_data.issue_id,
        "titulo": issue_data.tema,
        "prioridade": issue_data.prioridade,
        "area": issue_data.area,
        "responsavel": issue_data.responsavel,
        "criado_em": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "normas": ["ABNT NBR ISO/IEC 25010", "IEEE 830"],
    }
    yaml_lines = ["---"]
    for k, v in payload.items():
        if isinstance(v, list):
            yaml_lines.append(f"{k}:")
            yaml_lines.extend([f"  - {item}" for item in v])
        else:
            yaml_lines.append(f"{k}: \"{v}\"")
    yaml_lines.append("---")
    return "\n".join(yaml_lines)


def create_issue_template(issue_id: str, tema: str, prioridade: str, area: str) -> Path:
    issue_data = IssueData(issue_id=issue_id, tema=tema, prioridade=prioridade, area=area)
    validate_metadata(issue_data)
    RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    existing = sorted(RELATORIOS_DIR.glob(f"{issue_id}-*.md"))
    if existing:
        raise FileExistsError(f"ISSUE já existe: {existing[0]}")

    slug = slugify(tema)
    target = RELATORIOS_DIR / f"{issue_id}-{slug}.md"
    content = f"""{generate_yaml_frontmatter(issue_data)}
# {issue_id}: {tema}

## Objetivo
Descrever a implementação de forma orientada à execução por IA.

## Contexto Técnico
Contextualize dependências, componentes e motivação.

## Análise Técnica
Detalhe passos de análise para IA.

## Requisitos Funcionais
- [ ] RF-01:

## Requisitos Não-Funcionais
- [ ] RNF-01: Performance
- [ ] RNF-02: Segurança

## Artefatos Esperados
- Código
- Testes
- Documentação

## Critérios de Aceite
- [ ] Todos os testes passam.
- [ ] Links válidos.

## Riscos e Mitigações
- Risco:
- Mitigação:

## PROMPT:EXECUTAR
```yaml
objetivo: \"{tema}\"
issue_id: \"{issue_id}\"
prioridade: \"{prioridade}\"
area: \"{area}\"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | A definir |
"""
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    content += f"\n\n---\nChecksum: `{checksum}`\n"
    target.write_text(content, encoding="utf-8")

    log_payload = {
        "evento": "create_issue",
        "issue": issue_id,
        "arquivo": str(target),
        "checksum": checksum,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (LOGS_DIR / f"{issue_id}-create.log.json").write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--issue-id", required=True)
    parser.add_argument("--tema", required=True)
    parser.add_argument("--prioridade", required=True)
    parser.add_argument("--area", required=True)
    args = parser.parse_args()
    path = create_issue_template(args.issue_id, args.tema, args.prioridade, args.area)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
