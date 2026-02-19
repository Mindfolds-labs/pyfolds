# README — Auditoria de Ordem de Execução Neuronal (MPJRD Advanced)

Implementação consolidada do plano de correção da ordem de execução:

- Refratário mantém autoridade final do spike.
- SFA aplicada antes do threshold no `core/neuron.py`.
- bAP acoplado ao cálculo de `v_dend` (controlado por `backprop_enabled`).
- STDP com fonte de entrada configurável (`stdp_input_source = raw|stp`).
- LTD configurável (`ltd_rule = classic|current`).
- Semântica temporal unificada (`time_counter` incrementado no final do passo no backprop).

## Flags de compatibilidade

- `stdp_input_source="stp"` (default, retrocompatível)
- `ltd_rule="current"` (default, retrocompatível)

## Objetivo

Manter fidelidade biológica sem quebrar API existente, com testes determinísticos por mecanismo.
