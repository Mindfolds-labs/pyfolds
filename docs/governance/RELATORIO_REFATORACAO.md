# RELATÓRIO DE REFATORAÇÃO — Operação "Raiz Organizada"

## Backup
- Branch de backup criada: `backup/pre-refactor-20260216`
- Commit de snapshot: `chore(docs): snapshot pré-refatoração da estrutura de diretórios.`

## Comandos de referência (mv/rename)
```bash
mv RISK_REGISTER.md docs/governance/RISK_REGISTER.md
mv FOLD_QUALITY_PLAN.md docs/governance/QUALITY_ASSURANCE.md
mv SUMARIO_COMPLETO.md docs/governance/MASTER_PLAN.md

# Guias
mkdir -p docs/public/guides
mv docs/guide/* docs/public/guides/
mv docs/guides/* docs/public/guides/

# Desenvolvimento
mkdir -p docs/development
mv docs/developments/* docs/development/

# Qualidade/Governança
mkdir -p docs/governance/quality
mv docs/review/* docs/governance/quality/
mv docs/reviewer/* docs/governance/quality/
mv docs/reviews/* docs/governance/quality/

# Pesquisa
mkdir -p docs/research
mv docs/theory/* docs/research/

# Arquitetura
mkdir -p docs/architecture/specs docs/architecture/blueprints
mv docs/spec/* docs/architecture/specs/
mv docs/blueprint/* docs/architecture/blueprints/
mv docs/diagrams/* docs/architecture/blueprints/

# ADR
mkdir -p docs/governance/adr
mv docs/adr/* docs/governance/adr/
mv docs/development/adr/* docs/governance/adr/
```

## Resumo das ações executadas
- Pastas duplicadas consolidadas: `guide + guides`, `development + developments`, `review + reviewer + reviews`, `theory + research`, `blueprint + diagrams`, `spec` e `adr`.
- Documentos de governança da raiz migrados para `docs/governance/` com padronização de nomes.
- ADRs centralizados em `docs/governance/adr/` e renumerados em série (`ADR-001`...`ADR-034`).
- Criado índice histórico de ADRs: `docs/governance/adr/INDEX.md`.
- Criado hub de controle de execução: `docs/development/HUB_CONTROLE.md`.
- `README.md` da raiz reduzido ao padrão: Visão Geral, Instalação Rápida, Exemplo e Mapa de Documentação.
- `docs/README.md` atualizado para refletir a nova taxonomia.

## Links de governança
- HUB de controle: `docs/development/HUB_CONTROLE.md`
- Índice ADR: `docs/governance/adr/INDEX.md`
