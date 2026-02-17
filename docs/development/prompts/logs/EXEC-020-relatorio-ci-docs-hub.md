# EXEC-020 — Relatório CI Docs Hub e correções para Sphinx/MyST

## Status
✅ Concluída

## 1) Sumário executivo
O gate `docs-hub-quality` falha de forma recorrente porque combina `sphinx-build -W` (warning vira erro) com baseline documental ainda inconsistente (links internos históricos quebrados em ADRs e referência fora do source root em `docs/index.md`). Além disso, há dependência operacional de PlantUML sem garantia explícita de runtime Java/PlantUML em todos os ambientes locais.

## 2) Diagnóstico técnico

### 2.1 Causas-raiz (root causes)
1. **Referência externa ao source root no índice principal**
   - Arquivo: `docs/index.md`
   - Evidência: link `../README.md` na seção de referências.
   - Efeito: em builds Sphinx/MyST estritos, referências para fora de `docs/` são propensas a warning/erro de consistência de navegação.

2. **Links internos quebrados em ADRs legadas**
   - Escopo: `docs/governance/adr/*.md`
   - Evidência: varredura local encontrou 19 links relativos que apontam para arquivos inexistentes (ex.: `./ADR-004-validacao-multicamada.md`, `./ADR-006-safe-weight-law.md`).
   - Efeito: ruído de validação e potencial `myst.xref_missing`/link unresolved no build estrito.

3. **Dependência de PlantUML sem validação explícita de runtime**
   - Arquivos: `docs/conf.py`, `docs/development/diagrams/sequencia.md`, `docs/development/diagrams/componentes.md`.
   - Evidência: extensão `sphinxcontrib.plantuml` ativa e blocos `{plantuml}` presentes.
   - Efeito: se Java/PlantUML não estiver disponível, build pode falhar.

4. **Formato misto de fence para diagramas em Markdown de apoio**
   - Arquivo: `docs/development/diagrams/STYLE_GUIDE.md`
   - Evidência: uso de fences ` ```plantuml ` e ` ```mermaid ` (sem diretiva MyST `{plantuml}`/`{mermaid}`).
   - Efeito: risco de comportamento inconsistente entre renderizadores.

### 2.2 Arquivos/links mais críticos para `myst.xref_missing`
- `docs/governance/adr/ADR-022-adr-006-safe-weight-law-clamp-valida-o-num-rica.md` → `./ADR-004-validacao-multicamada.md`
- `docs/governance/adr/ADR-034-adr-009-testes-de-propriedades-matem-ticas-property-based.md` → `./ADR-006-safe-weight-law.md`, `./ADR-007-monitoramento-de-invariantes.md`, `./ADR-008-homeostase-com-anti-windup.md`
- `docs/governance/adr/ADR-017-adr-004-valida-o-multicamada-para-leitura-segura.md` → `./ADR-001-formato-binario-fold-mind.md`, `./ADR-002-compressao-zstd-por-chunk.md`, `./ADR-003-ecc-opcional-por-chunk.md`

## 3) Ajustes mínimos aplicados no ciclo
1. **Correção de referência no índice principal**
   - `docs/index.md`: substituído `../README.md` por `README.md` para manter a referência dentro de `docs/`.

2. **Governança do ciclo ISSUE/EXEC/HUB**
   - ISSUE criada: `ISSUE-020`.
   - Registro adicionado ao `execution_queue.csv`.
   - HUB sincronizado via `tools/sync_hub.py`.

## 4) Patch plan mínimo recomendado (próximos commits)
1. **ADR links cleanup (prioridade alta)**
   - Atualizar links legados para os nomes canônicos atuais (prefixados com numeração nova), reduzindo warnings de xref.

2. **PlantUML hardening no CI (prioridade alta)**
   - Garantir instalação/descoberta de PlantUML e Java no workflow docs.
   - Alternativa temporária: condicionar extensão PlantUML quando runtime indisponível.

3. **Padronização de fences (prioridade média)**
   - Converter fences `plantuml/mermaid` de guia para diretivas MyST (` ```{plantuml} ` / ` ```{mermaid} `).

## 5) Comandos executados
- `python -m sphinx -b html -W --keep-going docs docs/_build/html` (falhou localmente: Sphinx não instalado)
- `pip install -r requirements-docs.txt` (falhou localmente: ambiente com proxy restrito)
- `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-020-relatorio-ci-docs-hub-sphinx-myst.md`
- `python tools/sync_hub.py`
- `python tools/sync_hub.py --check`
- `python tools/check_issue_links.py docs/development/prompts/relatorios`

## 6) Critérios de aceite — status
- ✅ ISSUE criada no padrão e validada.
- ✅ EXEC criada com diagnóstico e plano mínimo.
- ✅ `execution_queue.csv` atualizado.
- ✅ `HUB_CONTROLE.md` sincronizado no mesmo ciclo.
- ⚠️ Build Sphinx local não executado por limitação de rede/proxy para instalar dependências.
