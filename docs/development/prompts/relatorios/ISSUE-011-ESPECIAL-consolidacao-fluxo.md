# ISSUE-011-ESPECIAL: Consolidação de Fluxo e Correção de Cards/Links

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex (Engenharia de Documentação) |
| Tipo | GOVERNANCE + DOCUMENTATION |
| Status | ⏳ Planejada para Execução |
| Prioridade | CRÍTICA |

## 🎯 Objetivo
Consolidar o fluxo operacional de issues (**CRIAR → ANALISAR → EXECUTAR → FINALIZAR**), corrigir cards/links faltantes no `HUB_CONTROLE.md`, validar status de todas as issues anteriores e estabelecer estrutura canônica para ISSUE-011 em diante.

## 📋 Escopo

### 2.1 Inclui
- ✅ Análise de status das ISSUE-001 até ISSUE-010-ESPECIAL
- ✅ Correção de cards não-gerados no `HUB_CONTROLE.md`
- ✅ Validação e correção de links em `docs/development/prompts/README.md`
- ✅ Atualização de status correto em `execution_queue.csv`
- ✅ Sincronização do HUB
- ✅ Validação de links e sintaxe em toda documentação
- ✅ Criação de LOG executável para ISSUE-011
- ✅ Finalizar issues pausadas que devem estar concluídas

### 2.2 Exclui
- ❌ Mudanças em `src/` (código-fonte)
- ❌ Alterações em testes (fora do escopo de docs)
- ❌ Reestruturação radical de docs (apenas ajustes operacionais)

## 📊 Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| Relatório ISSUE-011 | `docs/development/prompts/relatorios/ISSUE-011-ESPECIAL-consolidacao-fluxo.md` | Especificação de execução |
| Log de execução | `docs/development/prompts/logs/ISSUE-011-ESPECIAL-consolidacao-fluxo-LOG.md` | Evidência de execução |
| HUB_CONTROLE atualizado | `docs/development/HUB_CONTROLE.md` | Cards corrigidos e tabela consolidada |
| Fila de execução | `docs/development/execution_queue.csv` | Status atualizado para todas as issues |
| Validação de links | `docs/` e `README.md` | Links corrigidos |

## ✅ Critérios de Aceite
- [ ] `HUB_CONTROLE.md` com cards corretos para ISSUE-001..ISSUE-011
- [ ] `prompts/README.md` com índice de relatórios clicável
- [ ] `execution_queue.csv` refletindo status final consolidado
- [ ] `python tools/check_links.py docs/ README.md` sem erros críticos
- [ ] `python tools/sync_hub.py --check` verde
- [ ] `python -m compileall src/` sem `SyntaxError`
- [ ] ISSUE-011 registrada e rastreável em CSV + HUB
- [ ] LOG da execução com passos e evidências

## 📝 PROMPT:EXECUTAR
```yaml
fase: CONSOLIDACAO_FLUXO_COMPLETO
prioridade: CRITICA
responsavel: CODEX
dependente: [ISSUE-010, ISSUE-010-ESPECIAL]

acoes_imediatas:
  - task: "Validar status atual de todas as issues 001-010"
    comando: "grep -E 'Concluída|Pausada|Planejada' docs/development/execution_queue.csv"

  - task: "Gerar cards faltando em HUB_CONTROLE.md para ISSUE-005, 007, 008, 009"
  - task: "Adicionar índice de links em prompts/README.md"
  - task: "Corrigir status final de ISSUE-005, 007, 008, 009 em execution_queue.csv"
  - task: "Validar todos os links em docs/"
    comando: "python tools/check_links.py docs/ README.md"

  - task: "Sincronizar HUB e validar consistência"
    comando: "python tools/sync_hub.py && python tools/sync_hub.py --check"

  - task: "Validar sintaxe Python"
    comando: "python -m compileall src/"

  - task: "Registrar ISSUE-011 na fila"
  - task: "Criar LOG de execução"
```
