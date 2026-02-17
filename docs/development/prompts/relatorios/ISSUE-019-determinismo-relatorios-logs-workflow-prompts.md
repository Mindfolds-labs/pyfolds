# ISSUE-019: determinismo-relatorios-logs-workflow-prompts

## Metadados
- id: ISSUE-019
- tipo: GOVERNANCE
- titulo: Determinismo de relatórios e logs no workflow de prompts
- criado_em: 2026-02-17
- owner: Codex
- status: TODO

## 1. Objetivo
Aprimorar o workflow de documentação sem remover o conteúdo existente, tornando a execução de prompts mais determinística para qualquer agente (IA ou humano), com foco em:
- estrutura de relatório previsível,
- log técnico com erro encontrado + correção aplicada,
- rastreabilidade de arquivos afetados,
- validação operacional reproduzível.

## 2. Escopo

### 2.1 Inclui:
- Análise do fluxo atual de desenvolvimento em `docs/development/prompts/*`, `execution_queue.csv` e `HUB_CONTROLE.md`.
- Identificação de pontos de não determinismo no preenchimento de relatório e log de execução.
- Proposta de melhoria incremental (sem apagar o que já existe) para padronizar descrição de erro, impacto e correção.
- Definição de checklist mínimo para permitir que qualquer agente manipule os arquivos e siga as regras do workflow.
- Geração de artefatos ISSUE/EXEC da ISSUE-019 e sincronização HUB.

### 2.2 Exclui:
- Remoção de seções históricas já existentes em relatórios anteriores.
- Refatorações de código-fonte do pacote em `src/`.
- Alteração de lógica de negócio em ferramentas fora do domínio de governança documental.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-019-determinismo-relatorios-logs-workflow-prompts.md`
- `docs/development/prompts/execucoes/EXEC-019-determinismo-relatorios-logs-workflow-prompts.md`
- `docs/development/execution_queue.csv`
- `docs/development/HUB_CONTROLE.md`

## 4. Riscos
- Risco: reforçar regras sem exemplo operacional pode manter ambiguidades.
  Mitigação: registrar nesta ISSUE diagnóstico objetivo com "erro encontrado" e "correção proposta" por arquivo.
- Risco: divergência entre CSV e HUB no momento da entrega.
  Mitigação: execução obrigatória de `python tools/sync_hub.py` e validação `--check` no fechamento.

## 5. Critérios de Aceite
- ISSUE-019 criada em formato válido para `tools/validate_issue_format.py`.
- EXEC-019 descreve erros encontrados, impactos e correções propostas com foco em determinismo.
- `execution_queue.csv` atualizado com ISSUE-019.
- `HUB_CONTROLE.md` atualizado no mesmo commit após `python tools/sync_hub.py`.
- Validações executadas: formato ISSUE, links de ISSUE e sincronização HUB.

## 6. PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-019"
tipo: "GOVERNANCE"
titulo: "Determinismo de relatórios e logs no workflow de prompts"

passos_obrigatorios:
  - "Ler docs/development/execution_queue.csv"
  - "Confirmar próximo ISSUE-NNN regular"
  - "Criar ISSUE-019-determinismo-relatorios-logs-workflow-prompts.md"
  - "Criar EXEC-019-determinismo-relatorios-logs-workflow-prompts.md"
  - "Registrar ISSUE-019 no execution_queue.csv"
  - "Rodar python tools/sync_hub.py"
  - "Garantir alteração de docs/development/HUB_CONTROLE.md no mesmo commit"

validacao:
  - "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-019-determinismo-relatorios-logs-workflow-prompts.md"
  - "python tools/sync_hub.py --check"
  - "python tools/check_issue_links.py docs/development/prompts/relatorios"
```
