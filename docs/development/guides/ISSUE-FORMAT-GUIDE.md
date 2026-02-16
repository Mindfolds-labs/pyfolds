# Guia de Implementação — Formato Padrão de ISSUE para IA

## Objetivo
Este guia define como preencher corretamente o template canônico de ISSUE para execução por IA, reduzindo ambiguidades e permitindo validação automática.

## Estrutura obrigatória
Toda issue em `docs/development/prompts/relatorios/` deve seguir:

1. `# ISSUE-[NNN]: [Título]`
2. `## Metadados`
3. `## 1. Objetivo`
4. `## 2. Escopo`
   - `### 2.1 Inclui:`
   - `### 2.2 Exclui:`
5. `## 3. Artefatos Gerados`
6. `## 4. Riscos`
7. `## 5. Critérios de Aceite`
8. `## 6. PROMPT:EXECUTAR` (YAML obrigatório)

## Convenções de nomenclatura
- Arquivo: `ISSUE-[NNN]-[slug].md`
- `NNN`: sequencial com 3 dígitos (`001`, `002`, ..., `999`)
- `slug`: kebab-case, minúsculo, preferencialmente até 50 caracteres
- Local: `docs/development/prompts/relatorios/`

## Regras de escrita
- Objetivo: curto e verificável.
- Escopo: separar explicitamente o que entra e o que não entra.
- Artefatos: sempre citar caminho exato de cada arquivo.
- Riscos: incluir mitigação prática.
- Critérios de aceite: itens marcáveis (`- [ ]`) e testáveis.

## Bloco PROMPT:EXECUTAR
O bloco deve ser YAML válido dentro de cerca tripla:

```yaml
fase: NOME
prioridade: ALTA
responsavel: CODEX
dependente: [ISSUE-001]

acoes_imediatas:
  - task: "Descrever ação"
    output: "caminho/arquivo"
    prazo: "1h"

validacao_automatica:
  - tipo: "formato"
    ferramenta: "tools/validate_issue_format.py"
    criterio: "Passar sem erros"

pos_execucao:
  - atualizar: "docs/development/execution_queue.csv"
```

## Validação recomendada antes de abrir PR
```bash
python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-*.md
python tools/check_issue_links.py docs/development/prompts/relatorios
python tools/sync_hub.py --check
```

## Exemplo de uso
Use `docs/development/templates/ISSUE-IA-TEMPLATE.md` como ponto de partida para toda nova issue.
