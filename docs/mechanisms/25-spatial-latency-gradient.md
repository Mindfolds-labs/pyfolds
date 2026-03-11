# Title
Spatial Neural Latency Gradient

## Abstract
Modelo opcional de atraso espacial adiciona latência dependente de coordenada neural para simular propagação regional.

## Background
Atrasos regionais (~100 ms) aparecem em respostas corticais de envelope.

## Related neuroscience literature
- Crosse et al. (2016), atrasos no tracking de fala por EEG.
- Keitel et al. (2018), dinâmica espaço-temporal em córtex auditivo.

## Computational translation
`latency_kernel(neuron.coord)` fornece atraso adicional em ms.

## Implementation details
- Kernel baseado em norma espacial com saturação por `tanh`.
- Aplicado à latência em `WaveDynamicsMixin` quando habilitado.

## Files modified
- `src/pyfolds/advanced/speech_tracking.py`
- `src/pyfolds/advanced/wave.py`
- `src/pyfolds/core/config.py`

## Activation flags
- `enable_spatial_latency_gradient`
- `spatial_latency_max_ms`
- `spatial_latency_scale`

## Limitations
Kernel isotrópico simples; não representa anisotropia anatômica.

## Future work
Latência guiada por malha cortical e conectividade estrutural.
