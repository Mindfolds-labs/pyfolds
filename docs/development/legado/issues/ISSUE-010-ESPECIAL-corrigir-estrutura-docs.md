# ISSUE-010-ESPECIAL: Corrigir estrutura `docs/` — remover soltos e órfãos

## Metadados

| Campo | Valor |
|-------|-------|
| **Data** | 2026-02-17 |
| **Autor** | Codex (Engenharia de Documentação) |
| **Tipo** | GOVERNANCE |
| **Dependência** | ISSUE-011 (preparação) |
| **Issue relacionada** | ISSUE-010 |

## 1. Objetivo
Normalizar a raiz de `docs/` removendo arquivos soltos e órfãos **somente quando existir equivalente já consolidado em subpasta** e sem perda de conteúdo.

## 2. Justificativa
A raiz de `docs/` ainda contém artefatos legados (ex.: `quickstart.md`, `API_REFERENCE.md`) e histórico de remoções de órfãos que precisam ser validados antes da ISSUE-011.

## 3. Escopo

### 3.1 Inclui
- Verificar duplicação/migração de:
  - `ALGORITHM.md`, `SCIENTIFIC_LOGIC.md`, `METHODOLOGY.md`, `TEST_PROTOCOL.md` em `docs/science/`
  - `API_REFERENCE.md` em `docs/api/`
  - `installation.md` e `quickstart.md` em `docs/_quickstart/`
  - `BENCHMARKS.md` em `docs/benchmarks/` ou `docs/assets/`
- Verificar os 6 órfãos legados:
  - `ARCHITECTURE_V3.md`
  - `SCIENTIFIC_BASIS_V3.md`
  - `PHASE_CODING.md`
  - `WAVELENGTH_MAPPING.md`
  - `COOPERATIVE_INTEGRATION.md`
  - `FOLD_SPECIFICATION.md`
- Executar validação de links com `python tools/check_links.py`.
- Sincronizar HUB com `python tools/sync_hub.py` + `--check`.
- Registrar log completo em `docs/development/prompts/logs/ISSUE-010-ESPECIAL-corrigir-estrutura-docs-LOG.md`.

### 3.2 Exclui
- Não alterar conteúdo de arquivos.
- Não alterar `src/`.
- Não criar novas pastas.
- Não remover arquivo sem equivalente migrado/validado.

## 4. Critério de segurança para remoção
Só remover arquivo da raiz se:
1. houver equivalente em subpasta alvo;
2. o equivalente preservar conteúdo e referências críticas;
3. busca de referências não indicar dependência exclusiva do arquivo de raiz.

Se houver divergência de conteúdo, **NÃO remover**; consolidar em etapa dedicada.

## 5. Resultado desta execução (ISSUE-010-ESPECIAL)
- **Não houve remoções nesta rodada** por ausência de equivalentes canônicos nos diretórios-alvo definidos para os arquivos soltos.
- Os 6 órfãos listados já estavam removidos da raiz (`docs/`) desde execução anterior (ISSUE-010); apenas registros em subpastas temáticas persistem (`docs/research/wave/PHASE_CODING.md` e `docs/architecture/specs/FOLD_SPECIFICATION.md`).
- Estrutura registrada e pronta para próxima etapa de migração controlada antes da ISSUE-011.

## 6. PROMPT:EXECUTAR

```yaml
fase: GOVERNANCE_DOCS_NORMALIZACAO
prioridade: ALTA
responsavel: CODEX
issue: ISSUE-010-ESPECIAL

acoes:
  - "Inventariar arquivos soltos em docs/ e comparar com destinos-alvo"
  - "Checar órfãos históricos e confirmar status"
  - "Remover somente arquivos com duplicata canônica comprovada"
  - "Validar links após mudanças"
  - "Sincronizar HUB e fila de execução"
  - "Registrar log técnico completo"

regras_de_bloqueio:
  - "Se conteúdo divergir, não remover"
  - "Se arquivo não existir no destino-alvo, não remover"
  - "Sem alterações em src/"

validacao:
  - "python tools/check_links.py"
  - "python tools/sync_hub.py --check"
```
