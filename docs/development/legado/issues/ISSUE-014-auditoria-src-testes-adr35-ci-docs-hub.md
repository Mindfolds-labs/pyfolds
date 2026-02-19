# ISSUE-014: Auditoria SRC/Testes ADR-035 + gate CI de documentação estrutural

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex |
| Tipo | CODE |
| Prioridade | Crítica |

## 1. Objetivo
Executar a fase `AUDITORIA_SRC_TESTES_ADR35` com evidências atualizadas e consolidar integração de validação documental com **Sphinx + PyData Theme + MyST** em etapa de pre-build/CI.

## 2. Ações executadas
- `coletar_ambiente`
- `compilar_src`
- `smoke_import`
- `testar_packaging_editavel`
- `rodar_validacoes_docs_hub`
- `executar_testes`
- `consolidar_achados`
- `criar_adr_035` (atualização)
- `atualizar_fila_execucao`

## 3. Artefatos e evidências
- Logs: `docs/development/prompts/logs/ISSUE-014/*`
- Auditor de docs hub: `tools/docs_hub_audit.py`
- Artefatos automáticos:
  - `docs/development/generated/dependency_diagram.md`
  - `docs/development/generated/functions_table.md`
  - `docs/development/generated/metrics.md`
  - `docs/development/generated/metrics.json`
  - `docs/development/generated/module_inventory.md`
  - `docs/development/generated/api_structure_manifest.json`

## 4. Achados consolidados

### P0
- `pip install -e .` com build isolation continua falhando em rede restrita/proxy (`setuptools>=61.0` indisponível).

### P1
- Sem falhas funcionais na suíte de testes (`198 passed`).

### P2
- Warnings persistentes em marker `performance` e warning controlado de cleanup `mmap`.

## 5. Decisão técnica aplicada
Foi criado um **gate de qualidade documental estrutural** com falha de CI para:
1. Função/classe pública sem docstring.
2. Módulo público não referenciado no inventário automático.
3. Divergência estrutural entre código e artefatos gerados (manifesto/tabelas/diagrama/métricas).

Também foi adicionado build Sphinx (`-W`) no workflow de validação, garantindo consistência com stack MyST + PyData.
