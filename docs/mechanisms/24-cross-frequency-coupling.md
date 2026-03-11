# Title
Theta–Gamma Phase-Amplitude Coupling

## Abstract
Implementa métrica opcional de acoplamento fase-amplitude para estimar interação entre fase lenta e amplitude rápida.

## Background
PAC é frequentemente usado para caracterizar coordenação multiescala em processamento de fala.

## Related neuroscience literature
- Canolty & Knight (2010), PAC em cognição.
- Hyafil et al. (2015), cross-frequency em speech parsing.

## Computational translation
`compute_phase_amplitude_coupling(phase_theta, amp_gamma)` retorna MI e perfil por bins de fase.

## Implementation details
- Modulation Index (Tort-style KL normalizado).
- Executado apenas quando `enable_cross_frequency_coupling=True`.

## Files modified
- `src/pyfolds/advanced/speech_tracking.py`
- `src/pyfolds/advanced/wave.py`
- `src/pyfolds/core/config.py`

## Activation flags
- `enable_cross_frequency_coupling`

## Limitations
Estimativa simplificada por lote único.

## Future work
Estimativa PAC em janela móvel e testes com surrogates.
