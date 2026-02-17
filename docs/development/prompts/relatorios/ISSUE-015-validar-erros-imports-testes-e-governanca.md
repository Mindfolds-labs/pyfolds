# ISSUE-015: Validar erros, importação do pyfolds e governança de execução

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex |
| Tipo | CODE |
| Prioridade | Crítica |
| Justificativa | Garantir que as correções semânticas recentes não introduziram regressões e que o fluxo de importação/testes está estável. |

## 1. Objetivo
Executar validação completa de qualidade para o estado atual do repositório após as correções de erros semânticos, confirmando importação funcional de `pyfolds`, execução da suíte de testes e rastreabilidade por issue/ADR.

## 2. Escopo

### 2.1 Inclui:
- ✅ Verificação de compilação e smoke import de `pyfolds`.
- ✅ Execução da suíte de testes viável no ambiente.
- ✅ Registro formal da execução em issue e ADR.

### 2.2 Exclui:
- ❌ Alterações de feature/arquitetura fora de hardening e governança.

## 3. Artefatos Gerados

| Artefato | Localização | Descrição | Formato |
|---|---|---|---|
| Issue de execução | `docs/development/prompts/relatorios/ISSUE-015-validar-erros-imports-testes-e-governanca.md` | Planejamento e checklist da execução | `.md` |
| ADR de governança | `docs/governance/adr/ADR-036-governanca-validacao-integral-import-testes.md` | Decisão sobre gate de validação integral | `.md` |
| Fila atualizada | `docs/development/execution_queue.csv` | Registro da issue na fila operacional | `.csv` |

## 4. Riscos

| ID | Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|---|
| R01 | Divergência entre ambiente local e CI para import/editable install | Média | Alto | Executar `pip install -e . --no-build-isolation` e smoke import explícito |
| R02 | Regressão silenciosa em módulos avançados | Baixa | Alto | Executar `pytest -q` com suite padrão e registrar resultado |

## 5. Critérios de Aceite
- [ ] Compilação de `src/` concluída sem erros.
- [ ] `import pyfolds` executado com sucesso após instalação editável.
- [ ] Testes padrão (`pytest -q`) concluídos sem falhas.
- [ ] Issue validada por `tools/validate_issue_format.py`.
- [ ] Links de issues válidos via `tools/check_issue_links.py`.

## 6. PROMPT:EXECUTAR

```yaml
fase: VALIDACAO_INTEGRAL_POS_CORRECAO
prioridade: CRITICA
responsavel: CODEX
dependente: [ISSUE-014]

acoes_imediatas:
  - task: "Compilar toda a pasta src para verificar sintaxe/imports"
    output: "logs de compileall"
    prazo: "0.5h"
    comando: "python -m compileall src"
  - task: "Validar instalação editável e importação do pacote"
    output: "logs de pip install e smoke import"
    prazo: "0.5h"
    comando: "python -m pip install -e . --no-build-isolation && python -c 'import pyfolds'"
  - task: "Executar suíte padrão de testes"
    output: "logs de pytest"
    prazo: "1h"
    comando: "pytest -q"

validacao_automatica:
  - tipo: "formato"
    ferramenta: "tools/validate_issue_format.py"
    criterio: "Issue deve passar sem erros"
  - tipo: "links"
    ferramenta: "tools/check_issue_links.py"
    criterio: "Sem links inválidos no diretório de relatórios"
  - tipo: "hub"
    ferramenta: "tools/sync_hub.py --check"
    criterio: "Hub consistente com estado atual"

pos_execucao:
  - atualizar: "docs/development/execution_queue.csv"
  - sincronizar: "python tools/sync_hub.py"
  - validar: "python tools/sync_hub.py --check"
```
