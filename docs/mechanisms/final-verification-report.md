# Verificação final dos mecanismos de neural speech tracking

## Mecanismos implementados
1. Extração de envelope da fala (Hilbert + Gammatone aproximado).
2. Detecção de eventos do envelope (acoustic edges).
3. Phase reset por evento acústico.
4. Cross-frequency coupling (theta phase x gamma amplitude).
5. Gradiente espacial de latência.
6. Rotina automática de análise (`analyze_mechanisms`).

## Status de ativação por padrão
Todos os mecanismos acima estão **desligados por padrão** para preservar baseline:
- `enable_speech_envelope_tracking = False`
- `enable_phase_reset_on_audio_event = False`
- `enable_cross_frequency_coupling = False`
- `enable_spatial_latency_gradient = False`

## Mecanismos experimentais
Todos são marcados como experimentais por dependerem de entrada de áudio e/ou coordenadas neurais opcionais.

## Verificações executadas
- Buffers: novo buffer `_latency_delay_steps` registrado em `WaveDynamicsMixin`.
- `state_dict`: contém buffers extras somente em modelos wave (compatível com registro de buffers do PyTorch).
- Toggles: testes unitários cobrem desligado/ligado.
- Baseline preservado: quando flags estão desligadas, payload extra não é injetado.
- Debug tools: `analyze_mechanisms` provê deltas de atividade, estabilidade, custo e conectividade.
- Documentação: mini-artigos em `docs/mechanisms/` e ADRs em `docs/adr/`.

## Impacto computacional estimado
- Envelope Hilbert: baixo (FFT única O(N log N)).
- Envelope Gammatone aproximado: médio (múltiplas bandas FFT/IRFFT).
- Event detection: baixo (diferença + threshold).
- Phase reset: baixo (operação vetorial `where`).
- PAC: baixo-médio (binning + agregação por bins).
- Spatial latency kernel: baixo (norma + tanh).
