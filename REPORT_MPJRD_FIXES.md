# REPORT MPJRD FIXES

## Resumo executivo
Foram corrigidos os bugs de ordem de execução e fidelidade computacional no `MPJRDNeuronAdvanced`: SFA agora atua antes do threshold, o refratário absoluto passou a ser inviolável, o bAP agora altera a computação dendrítica real, STDP lê `x` pré-STP, a regra LTD foi alinhada a Bi & Poo, e `time_counter` passou a incrementar no final do passo.

## Mini-Story Scrum
- **User Story**: Como pesquisador, quero que o neurônio respeite refratário absoluto e aplique bAP e STDP corretamente, para garantir fidelidade e estabilidade.
- **Definition of Done**:
  - Novos testes por mecanismo adicionados e passando.
  - `pytest -q` passando na suíte.
  - Relatório técnico de evidências e mapa Bug→Mudança→Teste.

## Mapa Bug → Mudança → Teste

| Bug | Mudança implementada | Teste(s) |
|---|---|---|
| #1 Refratário violado por adaptação pós-processamento | SFA movida para antes do threshold na base (`u_eff = u - I_adapt`); `AdaptationMixin.forward` não recalcula `spikes`; atualização de adaptação ocorre após spikes confirmados pelo refratário. | `tests/unit/neuron/test_refractory.py`, `tests/unit/neuron/test_adaptation_sfa.py` |
| #2 bAP no-op | `dendrite_amplification` aplicada em `v_dend` com `amp = clamp(1+amp, 1, backprop_max_gain)` antes da integração somática. | `tests/unit/neuron/test_backprop_bap.py` |
| #3 STDP usando `x_modulated` | `ShortTermDynamicsMixin` passa `_x_pre_stp`; `STDPMixin` usa `kwargs.pop('_x_pre_stp', x)` como fonte de pre-spike. | `tests/unit/neuron/test_stp_stdp_contracts.py` |
| #4 `time_counter` no início | Incremento movido para o final de `BackpropMixin.forward`. | `tests/unit/neuron/test_time_counter.py` |
| #5 LTD incorreta | Fórmula LTD alterada para depender de `pre_spikes` (`-A_minus * trace_post * ltd_mask * pre_spikes`). | `tests/unit/neuron/test_stp_stdp_contracts.py` |

## Evidências objetivas
- Antes: testes e integrações esperavam `u_adapted` pós-refratário; esse fluxo permitia sobrescrever `spikes` após bloqueio refratário.
- Agora: `spikes` finais permanecem bloqueados no refratário absoluto e SFA atua no `u` pré-threshold.
- Execução de validação principal:
  - `pytest -q` → **passou** (`262 passed, 2 skipped, 3 deselected`).
  - `ruff .` não suportado pelo binário local (subcomando inválido); `ruff check .` executado, mas reporta issues históricas fora do escopo desta mudança.

## Notas de compatibilidade e risco
- API pública preservada para uso de `MPJRDNeuronAdvanced`.
- Campo `u_adapted` deixou de ser emitido; substituído por contrato explícito (`u_raw`, `u_eff`, `u`) refletindo etapa pré-threshold.
- Risco residual baixo: ajustes focados em ordem/semântica temporal com cobertura unitária dedicada.
- Rollback simples: reverter commit único da feature branch.

## Próximos passos
1. Opcional: padronizar lint CI para `ruff check .` (ou alinhar comando requerido).
2. Adicionar teste de regressão temporal multi-step para delay bAP em 2–5 ms com valores controlados.
3. Expandir testes STDP para janelas Δt positivas/negativas explícitas (LTP/LTD com sequência temporal controlada).
