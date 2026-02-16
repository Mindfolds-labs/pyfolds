# ISSUE-009: Padronização de Formatos de ISSUEs para Interação com IA

## Metadados

| Campo | Valor |
|-------|-------|
| **Data** | 2026-02-16 |
| **Autor** | Codex (Engenharia de Documentação) |
| **Issue de Origem** | ISSUE-003 |
| **Normas de Referência** | IEEE 830, ISO/IEC 25010, Google Developer Documentation Style Guide |

## 1. Objetivo
Padronizar o formato de todos os relatórios de issue voltados à execução por IA no repositório, adotando um template canônico único derivado do padrão consolidado no ISSUE-003.
A meta é garantir consistência estrutural, executabilidade de prompts, rastreabilidade e validação automatizada no fluxo CRIAR → ANALISAR → EXECUTAR → FINALIZAR.

## 2. Escopo

### 2.1 Inclui:
- ✅ Definição de template obrigatório para ISSUEs de IA.
- ✅ Guia prático de uso e exemplos de preenchimento.
- ✅ Checklist de conformidade para revisão humana.
- ✅ Script de validação estrutural de arquivos ISSUE.
- ✅ Script de verificação de links/referências entre ISSUEs.
- ✅ Workflow de CI para validar novos ISSUEs automaticamente.
- ✅ Atualização da fila de execução e sincronização do HUB.

### 2.2 Exclui:
- ❌ Migração completa de 100% dos ISSUEs históricos nesta etapa.
- ❌ Alterações em políticas de governança fora do domínio de prompts/issues.
- ❌ Mudanças em código-fonte de runtime da biblioteca (`src/`).

## 3. Artefatos Gerados

| Artefato | Localização | Descrição | Formato |
|----------|-------------|-----------|---------|
| ISSUE canônica | `docs/development/prompts/relatorios/ISSUE-009-padronizacao-formatos-ia.md` | Relatório principal da padronização | `.md` |
| Template canônico | `docs/development/templates/ISSUE-IA-TEMPLATE.md` | Modelo oficial para novas ISSUEs | `.md` |
| Guia de implementação | `docs/development/guides/ISSUE-FORMAT-GUIDE.md` | Manual de adoção do padrão | `.md` |
| Checklist | `docs/development/checklists/ISSUE-VALIDATION.md` | Lista de verificação pré-PR | `.md` |
| Validador de formato | `tools/validate_issue_format.py` | Valida se ISSUEs seguem o padrão | `.py` |
| Verificador de links | `tools/check_issue_links.py` | Valida links e referências entre ISSUEs | `.py` |
| Inventário de conformidade | `docs/inventory/issues-conformity.csv` | Mapeia conformidade dos ISSUEs atuais | `.csv` |
| CI de validação | `.github/workflows/validate-issues.yml` | Executa validações em push/PR | `.yml` |
| Workflow integrado | `docs/development/WORKFLOW_INTEGRADO.md` | Fluxo operacional conectado com execução real | `.md` |

## 4. Riscos

| ID | Risco | Probabilidade | Impacto | Mitigação |
|----|-------|---------------|---------|-----------|
| R01 | Quebra de links em ISSUEs existentes durante padronização | Média | Alto | Executar `tools/check_issue_links.py` em toda alteração |
| R02 | Baixa adesão ao novo formato pela equipe | Média | Médio | Publicar template + checklist + validação automática no CI |
| R03 | Inconsistências em migração de ISSUEs legados | Baixa | Alto | Revisão humana + inventário de conformidade |
| R04 | Sobrecarga inicial de criação de ISSUEs | Alta | Baixo | Reuso de template e guia de preenchimento rápido |

## 5. Critérios de Aceite
- [ ] Existe template canônico em `docs/development/templates/ISSUE-IA-TEMPLATE.md`.
- [ ] Existe guia de uso em `docs/development/guides/ISSUE-FORMAT-GUIDE.md`.
- [ ] Existe checklist de validação em `docs/development/checklists/ISSUE-VALIDATION.md`.
- [ ] `tools/validate_issue_format.py` valida ISSUEs com retorno não-zero em erro.
- [ ] `tools/check_issue_links.py` detecta referências inválidas entre ISSUEs.
- [ ] Workflow `.github/workflows/validate-issues.yml` está configurado para PR/push.
- [ ] `docs/development/execution_queue.csv` e `docs/development/HUB_CONTROLE.md` refletem ISSUE-009.
- [ ] Há orientação operacional prática em `docs/development/prompts/README.md` e `docs/development/WORKFLOW_INTEGRADO.md`.

## 6. PROMPT:EXECUTAR

```yaml
fase: PADRONIZACAO_ISSUES_IA
prioridade: CRITICA
responsavel: CODEX
dependente: [ISSUE-003]

acoes_imediatas:
  - task: "Criar template canônico de ISSUE para IA"
    output: "docs/development/templates/ISSUE-IA-TEMPLATE.md"
    prazo: "2h"

  - task: "Publicar guia de adoção e boas práticas"
    output: "docs/development/guides/ISSUE-FORMAT-GUIDE.md"
    prazo: "2h"

  - task: "Criar checklist de validação pré-publicação"
    output: "docs/development/checklists/ISSUE-VALIDATION.md"
    prazo: "1h"

  - task: "Implementar validador automático de estrutura"
    output: "tools/validate_issue_format.py"
    prazo: "3h"

  - task: "Implementar verificador de links e referências entre ISSUEs"
    output: "tools/check_issue_links.py"
    prazo: "2h"

  - task: "Atualizar textos operacionais (prompts, contribuição e release)"
    output: "docs/development/prompts/README.md"
    prazo: "2h"

  - task: "Publicar workflow integrado da issue até execução real"
    output: "docs/development/WORKFLOW_INTEGRADO.md"
    prazo: "1h"

  - task: "Atualizar fila de execução e sincronizar HUB"
    output: "docs/development/execution_queue.csv"
    prazo: "30m"

validacao_automatica:
  - tipo: "formato"
    ferramenta: "python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-*.md"
    criterio: "Todas as ISSUEs novas passam no padrão obrigatório"

  - tipo: "links"
    ferramenta: "python tools/check_issue_links.py docs/development/prompts/relatorios"
    criterio: "Nenhuma referência para ISSUE inexistente"

  - tipo: "hub"
    ferramenta: "python tools/sync_hub.py --check"
    criterio: "Tabela resumida do HUB sincronizada com execution_queue.csv"

pos_execucao:
  - atualizar: "docs/development/execution_queue.csv"
    status: "Em progresso"

  - sincronizar: "docs/development/HUB_CONTROLE.md"
    comando: "python tools/sync_hub.py"

  - verificar: "consistência_hub"
    comando: "python tools/sync_hub.py --check"

  - notificar: "PR"
    mensagem: "ISSUE-009 implementada com template, guias, checklist e automações"
```
