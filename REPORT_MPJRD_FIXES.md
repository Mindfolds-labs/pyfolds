# REPORT_MPJRD_FIXES

## Resumo executivo
Foram corrigidos cinco problemas de fidelidade computacional no `MPJRDNeuronAdvanced` com foco na ordem de execução dos mixins e no contrato biológico dos mecanismos avançados. As correções garantem: (1) SFA antes do threshold e sem sobrescrever spikes pós-refratário, (2) bAP com efeito real na dinâmica dendrítica, (3) STDP lendo o pré-spike original antes do STP, (4) `time_counter` avançando no fim do passo, e (5) LTD clássico dependente de `pre_spikes`.

## Mini-Story Scrum
- **User Story:** Como pesquisador, quero que o neurônio respeite refratário absoluto e aplique bAP e STDP corretamente, para garantir fidelidade e estabilidade.
- **Definition of Done:** novos testes por mecanismo adicionados e passando, `pytest -q` verde, relatório e changelog atualizados.

## Mapa Bug → Mudança → Teste

Config flags adicionadas para retrocompatibilidade auditável:
- `stdp_input_source = "raw" | "stp"` (default `stp`).
- `ltd_rule = "classic" | "current"` (default `current`).

| Bug | Mudança implementada | Teste(s) de evidência |
|---|---|---|
| #1 Refratário absoluto violado por adaptação tardia | SFA agora é aplicada no `forward` da base antes do threshold via `_apply_sfa_before_threshold`; `AdaptationMixin.forward` não recalcula `spikes` e só atualiza corrente com spikes finais. | `tests/unit/neuron/test_refractory.py`, `tests/unit/neuron/test_adaptation_sfa.py` |
| #2 bAP no-op | `dendrite_amplification` passou a multiplicar `v_dend` no `MPJRDNeuron.forward` com clamp por `backprop_max_gain`. | `tests/unit/neuron/test_backprop_bap.py` |
| #3 STDP lia `x_modulated` | `ShortTermDynamicsMixin` passa `_x_pre_stp`; `STDPMixin.forward` consome e remove via `pop` e usa no detector de pré-spike. | `tests/unit/neuron/test_stp_stdp_contracts.py::test_stdp_uses_pre_stp_input_for_pre_spike_detection` |
| #4 `time_counter` no início | `BackpropMixin.forward` agora processa com tempo atual e incrementa `time_counter` apenas ao final. | `tests/unit/neuron/test_time_counter.py` |
| #5 LTD incorreta | `delta_ltd` ajustada para usar `pre_spikes` (regra Bi & Poo clássica). | `tests/unit/neuron/test_stp_stdp_contracts.py::test_stdp_ltd_depends_on_pre_spikes_not_post_spike_only` |

## Evidência objetiva
### Estado anterior (TDD vermelho)
Rodada inicial dos novos testes falhou com sintomas esperados:
- refratário/timing não respeitados,
- bAP sem efeito computacional,
- STDP pré-spike/LTD fora do contrato.

### Estado atual (TDD verde)
- `pytest -q tests/unit/neuron/test_refractory.py tests/unit/neuron/test_adaptation_sfa.py tests/unit/neuron/test_backprop_bap.py tests/unit/neuron/test_stp_stdp_contracts.py tests/unit/neuron/test_time_counter.py` → **6 passed**.
- `pytest -q` → **263 passed, 2 skipped, 3 deselected**.

## Compatibilidade e risco (ITIL)
- **Mudança de alto risco mitigada:** regras de spike/refratário e STDP alteradas com cobertura específica por mecanismo.
- **Compatibilidade API:** API pública preservada; ajustes ficaram em funções internas e no fluxo interno de mixins.
- **Rollback simples:** revert do commit único restaura comportamento anterior.

## Próximos passos
1. Adicionar testes de regressão temporal multi-step para janelas STDP longas.
2. Opcional: separar explicitamente `u_effective` no contrato de saída para telemetria científica.
3. Alinhar lint global do repositório para tornar `ruff check .` green sem débito legado.


## Comandos executados (evidência)
- `pytest tests/unit/advanced/test_refractory.py -q`
- `pytest tests/unit/advanced/test_adaptation.py -q`
- `pytest tests/unit/advanced/test_stdp.py -q`
- `pytest tests/integration/test_neuron_advanced.py -q`
- `pytest -q`
- `ruff check .` (falhas preexistentes fora do escopo).
