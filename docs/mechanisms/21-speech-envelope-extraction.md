# Title
Speech Envelope Extraction for Neural Tracking

## Abstract
Este mecanismo adiciona extração de envelope acústico por Hilbert e por banco de filtros gammatone para alimentar rotinas neurais dependentes de envelope.

## Background
Literatura de cortical tracking mostra que a escolha do envelope altera robustez de sincronização neural com fala.

## Related neuroscience literature
- Ding & Simon (2014), cortical entrainment to continuous speech.
- Gross et al. (2013), tracking dinâmico da fala por oscilações lentas.

## Computational translation
Função `extract_speech_envelope(audio, sample_rate, method)` retorna envelope, força de onsets e espectro de modulação.

## Implementation details
- Método `hilbert`: sinal analítico via FFT e magnitude.
- Método `gammatone`: aproximação ERB/gammatone em múltiplas bandas + envelope médio.
- Gatilho por `enable_speech_envelope_tracking`.

## Files modified
- `src/pyfolds/advanced/speech_tracking.py`
- `src/pyfolds/advanced/wave.py`
- `src/pyfolds/core/config.py`

## Activation flags
- `enable_speech_envelope_tracking`
- `speech_envelope_method`

## Limitations
Implementação gammatone aproximada (sem dependência externa dedicada).

## Future work
Adicionar backend opcional com filtros gammatone canônicos (Hohmann) e validação em corpus de fala real.
