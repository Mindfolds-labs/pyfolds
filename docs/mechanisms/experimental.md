# Mecanismos experimentais

Mecanismos classificados como experimentais por dependerem de toggles em `pyfolds.advanced.experimental` ou por estarem em trilhas científicas/avaliativas sem ativação padrão obrigatória.

## Toggles oficiais em `pyfolds.advanced.experimental`

- [Phase Gating Experiment](phase_gating_experiment.md) — controlado por `enable_phase_gating`.
- [Cross-Frequency Coupling](24-cross-frequency-coupling.md) — relacionado ao controle experimental de modulação/gating de fase.
- [Replay Pruning Consolidation](replay-pruning-consolidation.md) — relacionado ao toggle `enable_sleep_consolidation`.
- [Envelope Event Detection](22-envelope-event-detection.md) — associado a trilhas de análise dinâmica de eventos.
- [Engram Resonance Reporting](engram-resonance-reporting.md) — diagnóstico comparativo de mecanismos.

## Trilha de pesquisa (sem ativação padrão em mixins)

- [Speech Envelope Extraction](21-speech-envelope-extraction.md)
- [Phase Reset on Events](23-phase-reset-on-events.md)
- [Spatial Latency Gradient](25-spatial-latency-gradient.md)
- [Mechanisms Overview](overview.md)

## Referência de código

- `src/pyfolds/advanced/experimental.py`
- `src/pyfolds/advanced/speech_tracking.py`
