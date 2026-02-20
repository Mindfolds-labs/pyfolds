---
id: "ISSUE-011"
titulo: "Micro-otimização do forward e análise dos testes pulados"
prioridade: "Alta"
area: "Qualidade/Performance"
responsavel: "Codex"
criado_em: "2026-02-20"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
adr_vinculada: "ADR-043"
---
# ISSUE-011: Micro-otimização do forward e análise dos testes pulados

## Objetivo
Aplicar otimização segura no `forward` do neurônio (redução de overhead de alocação/escrita indexada) e executar auditoria dos 7 testes pulados com justificativa objetiva por ambiente.

## Contexto Técnico
No `MPJRDNeuron.forward`, a integração dendrítica usava alocação por `torch.zeros` seguida de escrita indexada em loop. A alteração substitui esse padrão por coleta de saídas e `torch.stack`, mantendo semântica e shape `[B, D]`.

## Análise Técnica
### Alteração aplicada (Quick Win)
- Antes:
  - `v_dend = torch.zeros(B, D, device=device)`
  - preenchimento por índice `v_dend[:, d_idx] = ...`
- Depois:
  - `dendrite_outputs = [...]`
  - `v_dend = torch.stack(dendrite_outputs, dim=1)`

### Resultado da suíte
Execução integral: `284 passed, 7 skipped, 4 deselected, 0 failed`.

### Justificativa dos 7 pulados
1. TensorFlow não instalado (4 testes): dependência opcional não presente no ambiente.
2. CUDA não disponível (1 teste): teste condicionado a GPU.
3. `cryptography` não instalada (1 teste): dependência opcional para cenário de serialização com assinatura.
4. Total consolidado: 7 skips esperados por marcadores/guards da suíte.

## Requisitos Funcionais
- [x] RF-01: Otimizar integração dendrítica sem alterar contrato funcional.
- [x] RF-02: Executar suíte completa de regressão.
- [x] RF-03: Explicar tecnicamente todos os skips.

## Requisitos Não-Funcionais
- [x] RNF-01: Preservar compatibilidade com CPU-only.
- [x] RNF-02: Manter rastreabilidade de execução por logs.
- [x] RNF-03: Sincronizar governança (fila/HUB).

## Artefatos Esperados
- [x] `src/pyfolds/core/neuron.py`
- [x] `outputs/test_logs/pytest_full.log`
- [x] `outputs/test_logs/pytest-junit.xml`
- [x] `docs/development/prompts/execucoes/EXEC-011-otimizacao-forward-e-analise-de-skips.md`

## Critérios de Aceite
- [x] O forward foi otimizado sem regressão funcional.
- [x] Suíte completa executada sem falhas.
- [x] Skips analisados e justificados tecnicamente.

## Riscos e Mitigações
- Risco: ganho de performance marginal em ambientes pequenos.
  - Mitigação: tratar como micro-otimização segura e incremental.
- Risco: skips serem confundidos com falhas.
  - Mitigação: manter relatório explícito com razão por dependência/hardware.

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | `src/pyfolds/core/neuron.py` |
| RF-02 | `outputs/test_logs/pytest_full.log` |
| RF-03 | `outputs/test_logs/pytest_full.log` (seção `short test summary info`) |

## PROMPT:EXECUTAR
```yaml
objetivo: "Micro-otimização do forward e análise dos testes pulados"
issue_id: "ISSUE-011"
prioridade: "Alta"
area: "Qualidade/Performance"
adr_vinculada: "ADR-043"
evidencias:
  - outputs/test_logs/pytest_full.log
  - outputs/test_logs/pytest-junit.xml
  - src/pyfolds/core/neuron.py
```
