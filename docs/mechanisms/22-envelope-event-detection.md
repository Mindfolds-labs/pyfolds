# Title
Acoustic Edge Detection from Speech Envelope

## Abstract
Detector de eventos por derivada positiva do envelope para identificar transientes acústicos relevantes.

## Background
Respostas evocadas por bordas acústicas são hipótese central para parte do tracking de fala.

## Related neuroscience literature
- Oganian & Chang (2019), speech envelope landmarks.
- Doelling et al. (2014), phase-locking por onsets.

## Computational translation
`detect_envelope_events(envelope)` retorna `onset_times`, `onset_strength` e `event_mask`.

## Implementation details
- Derivada temporal retificada.
- Threshold adaptativo: média + k*desvio.
- Saída pode alimentar reset de fase/gating/priorização.

## Files modified
- `src/pyfolds/advanced/speech_tracking.py`
- `src/pyfolds/advanced/wave.py`

## Activation flags
Ativado indiretamente por `enable_speech_envelope_tracking`.

## Limitations
Threshold único pode perder eventos em SNR muito baixo.

## Future work
Threshold adaptativo por janela e histerese de eventos.
