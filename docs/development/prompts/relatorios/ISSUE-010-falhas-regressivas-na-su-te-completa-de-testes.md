---
id: "ISSUE-010"
titulo: "Falhas regressivas na suíte completa de testes"
prioridade: "Alta"
area: "Qualidade"
responsavel: "Codex"
criado_em: "2026-02-20"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
adr_vinculada: "ADR-042"
---
# ISSUE-010: Falhas regressivas na suíte completa de testes

## Objetivo
Executar a suíte integral de testes com rastreabilidade completa, identificar falhas regressivas atuais e consolidar plano de remediação orientado a risco.

## Contexto Técnico
A suíte completa (`pytest tests -v`) foi executada após instalação de dependências e pacote local. Resultado consolidado:
- 291 testes selecionados
- 279 aprovados
- 5 falhos
- 7 pulados
- 11.49s de execução

Falhas concentram-se em:
1. integração/neuron advanced (ganho dendrítico com backprop desligado),
2. adaptation mixin com `inference_mode` string,
3. contrato de alias de import público legado.

## Análise Técnica
### Evidências produzidas
- `outputs/test_logs/pip_install.log`
- `outputs/test_logs/pytest_full.log`
- `outputs/test_logs/pytest-junit.xml`
- `outputs/test_logs/compileall.log`
- `outputs/test_logs/check_api_docs.log`
- `outputs/test_logs/check_links.log`

### Falhas capturadas
- `tests/integration/test_neuron_advanced.py::TestAdvancedNeuron::test_backprop_disabled_does_not_apply_dendritic_gain`
- `tests/unit/advanced/test_adaptation.py::TestAdaptationMixin::test_adaptation_respects_string_inference_mode`
- `tests/unit/advanced/test_adaptation.py::TestAdaptationMixin::test_forward_updates_u_for_downstream_mixins`
- `tests/unit/neuron/test_backprop_bap.py::test_bap_amplification_changes_dendritic_computation_and_clamps_gain`
- `tests/unit/test_public_import_surface.py::test_v1_aliases_emit_deprecation_warning_and_match_v2_targets_until_2_0`

## Requisitos Funcionais
- [x] RF-01: Executar suíte completa com log integral.
- [x] RF-02: Gerar issue operacional padronizada.
- [x] RF-03: Registrar ADR de governança para execuções integrais.
- [x] RF-04: Publicar relatório consolidado com métricas.

## Requisitos Não-Funcionais
- [x] RNF-01: Rastreabilidade de execução (artefatos persistidos).
- [x] RNF-02: Reprodutibilidade local (comandos explícitos).
- [x] RNF-03: Governança (vínculo com ADR e fila de execução).

## Artefatos Esperados
- [x] Documentação de issue.
- [x] ADR de governança.
- [x] Relatório final de execução.
- [x] Logs e XML de testes.

## Critérios de Aceite
- [x] Execução integral registrada.
- [x] Falhas identificadas nominalmente.
- [x] Links válidos.
- [x] Todos os testes passam (contexto CPU/Torch; skips opcionais mantidos).

## Riscos e Mitigações
- Risco: regressões continuarem abertas e bloquearem release.
  - Mitigação: priorizar correções por severidade (P0 comportamento funcional, P1 compatibilidade pública).
- Risco: perda de contexto entre execução e correção.
  - Mitigação: manter issue e relatório como fontes canônicas vinculadas à ADR-042.

## PROMPT:EXECUTAR
```yaml
objetivo: "Falhas regressivas na suíte completa de testes"
issue_id: "ISSUE-010"
prioridade: "Alta"
area: "Qualidade"
adr_vinculada: "ADR-042"
evidencias:
  - outputs/test_logs/pytest_full.log
  - outputs/test_logs/pytest-junit.xml
  - docs/RELATORIO_FINAL_EXECUCAO_TESTES_ISSUE-010.md
```


## Investigação Complementar (falhas + skips + treino)
- [x] Reexecução completa com motivos de skip (`pytest -rs`) registrada em `outputs/test_logs/pytest_full_rerun.log`.
- [x] Reexecução focada nos 5 testes falhos com stacktrace em `outputs/test_logs/pytest_failed_focus.log`.
- [x] Simulação de treinamento com MNIST (download + execução) para modelos `py` e `wave`, evidências em `outputs/test_logs/mnist_train_*.log`.
- [x] Relatório técnico detalhado publicado em `docs/RELATORIO_INVESTIGACAO_FALHAS_E_SKIPS_ISSUE-010.md`.

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | `outputs/test_logs/pytest_full.log` |
| RF-02 | Este arquivo (`ISSUE-010`) |
| RF-03 | `docs/governance/adr/ADR-042-governanca-de-execucao-integral-de-testes-e-dossie-de-qualidade.md` |
| RF-04 | `docs/RELATORIO_FINAL_EXECUCAO_TESTES_ISSUE-010.md` |


## Atualização pós-correção
- [x] Regressões corrigidas e validadas em `outputs/test_logs/pytest_full_fixed.log` (284 passed, 7 skipped).
- [x] Treino MNIST de 50 épocas executado para modelos `py` e `wave` com logs em `outputs/test_logs/mnist_train_*_50ep_console.log`.
