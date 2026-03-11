# Mecanismos estáveis

Mecanismos classificados como estáveis por estarem ativos na composição padrão de `pyfolds.advanced` (mixins e classes avançadas) ou por sustentarem a governança operacional dos experimentos.

## Mixins ativos em `pyfolds.advanced`

- [Backpropagation](../mechanisms/data-buffer-state-strategy.md) — ativo em `MPJRDNeuronAdvanced` e `MPJRDWaveNeuronAdvanced` via `BackpropMixin`.
- [Short-term dynamics](../mechanisms/data-buffer-state-strategy.md) — ativo via `ShortTermDynamicsMixin`.
- [STDP](../mechanisms/phase-activity-observability.md) — ativo via `STDPMixin`.
- [Adaptation (SFA)](../mechanisms/data-buffer-state-strategy.md) — ativo via `AdaptationMixin`.
- [Refractory](../mechanisms/debug-observability-playbook.md) — ativo via `RefractoryMixin`.
- [Wave dynamics](../mechanisms/phase-activity-observability.md) — disponível na cadeia via `WaveDynamicsMixin`/`CircadianWaveMixin` (ativação condicionada por `cfg.wave_enabled` e `cfg.circadian_enabled`).

## Mecanismos de infraestrutura em uso contínuo

- [Experimental Toggles](experimental_toggles.md) — infraestrutura estável para governar ablações A/B.
- [Connectivity and Pruning](connectivity-and-pruning.md) — mecanismo estrutural ativo no runtime.
- [Data Buffer State Strategy](data-buffer-state-strategy.md) — estratégia de estado para execução/telemetria.
- [Debug and Observability Playbook](debug-observability-playbook.md) — práticas operacionais de diagnóstico.
- [Final Verification Report](final-verification-report.md) — verificação operacional consolidada.
- [Phase Activity Observability](phase-activity-observability.md) — observabilidade de fase em produção.
- [Scientific Alignment Review](scientific-alignment-review.md) — rastreabilidade técnico-científica estável.

## Referência de código

- `src/pyfolds/advanced/__init__.py`
- `src/pyfolds/advanced/experimental.py`
