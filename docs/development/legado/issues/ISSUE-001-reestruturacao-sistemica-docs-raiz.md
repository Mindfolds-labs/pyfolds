# ISSUE-001: Reestruturação sistêmica de /docs e raiz (registro legado)

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-16 |
| Autor | Codex |
| Tipo | DOCS |
| Prioridade | Alta |

## 1. Objetivo
Registrar referência histórica da ISSUE-001 para integridade de links entre relatórios legados.

## 2. Escopo

### 2.1 Inclui:
- ✅ Registro mínimo da issue histórica para rastreabilidade.

### 2.2 Exclui:
- ❌ Reexecução técnica da issue original.

## 3. Artefatos Gerados

| Artefato | Localização | Descrição | Formato |
|---|---|---|---|
| Registro legado | `docs/development/prompts/relatorios/ISSUE-001-reestruturacao-sistemica-docs-raiz.md` | Documento de referência para validação de links | `.md` |

## 4. Riscos

| ID | Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|---|
| R01 | Divergência entre título histórico e registro resumido | Baixa | Baixo | Manter título idêntico ao histórico da fila |

## 5. Critérios de Aceite
- [ ] Arquivo de referência disponível para validação de links.

## 6. PROMPT:EXECUTAR

```yaml
fase: REGISTRO_LEGADO
prioridade: BAIXA
responsavel: CODEX
dependente: []

acoes_imediatas:
  - task: "Criar arquivo de referência histórica"
    output: "docs/development/prompts/relatorios/ISSUE-001-reestruturacao-sistemica-docs-raiz.md"
    prazo: "0.2h"

validacao_automatica:
  - tipo: "links"
    ferramenta: "tools/check_issue_links.py"
    criterio: "Referências de ISSUE-001 resolvidas"

pos_execucao:
  - atualizar: "docs/development/execution_queue.csv"
```
