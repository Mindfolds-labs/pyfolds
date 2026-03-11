# Title
Event-Driven Phase Reset

## Abstract
Quando eventos acústicos fortes são detectados, a fase local é realinhada para reduzir drift e melhorar sincronização.

## Background
Modelos de phase reset sugerem alinhamento por evento além de mera oscilação autônoma.

## Related neuroscience literature
- Giraud & Poeppel (2012), papel de oscilações em parsing de fala.
- Lakatos et al. (2008), reset de fase por estímulos relevantes.

## Computational translation
`reset_phase_if_event(phase, event_strength)` com threshold configurável.

## Implementation details
- Aplicado somente com `enable_phase_reset_on_audio_event`.
- Integrado na pipeline `WaveDynamicsMixin.forward` após extração de eventos.

## Files modified
- `src/pyfolds/advanced/speech_tracking.py`
- `src/pyfolds/advanced/wave.py`
- `src/pyfolds/core/config.py`

## Activation flags
- `enable_phase_reset_on_audio_event`
- `phase_reset_threshold`
- `phase_reset_target`

## Limitations
Reset global por lote; não modela topografia temporal fina.

## Future work
Reset por subpopulação neural e força proporcional ao evento.
