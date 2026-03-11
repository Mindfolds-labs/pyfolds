# Auditoria de Arquitetura Atual (canônica)

Este é o **documento oficial de referência** da auditoria arquitetural corrente do `pyfolds`.

## Fonte canônica da auditoria atual

- **Fonte canônica:** `docs/sheer-audit/sheerdocs/code_map.md`

## Por que esta fonte é a canônica

- `sheerdocs/code_map.md` é o artefato técnico consolidado e legível da varredura estrutural do repositório.
- `reports/` contém principalmente evidências operacionais de execução (logs/comandos), não o retrato arquitetural final.

## Evidências e materiais complementares

- Matriz de execução: `docs/sheer-audit/reports/sheer_execution_matrix.md`
- Comandos executados: `docs/sheer-audit/reports/commands_executed.md`
- Modelo de repositório (JSON): `docs/sheer-audit/sheerdocs/repo_model.json`
- UML de pacotes: `docs/sheer-audit/sheerdocs/uml/package.svg`
- UML de classes: `docs/sheer-audit/sheerdocs/uml/class_overview.svg`

## Regra de atualização

Sempre que uma nova auditoria substituir a corrente:

1. Atualizar a fonte canônica em `docs/sheer-audit/README.md`.
2. Atualizar este arquivo (`current_architecture_audit.md`) para apontar para a nova fonte.
3. Preservar os artefatos anteriores em `reports/` para rastreabilidade.
