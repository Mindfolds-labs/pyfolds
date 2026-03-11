# Cross-Frequency Coupling

## Objetivo
Medir acoplamento fase-amplitude (theta-gamma) como indicador de coordenação multiescala.

## Variáveis
- **Entrada:** `phase_theta`, `amp_gamma`.
- **Controle:** `enable_cross_frequency_coupling`.
- **Saída:** `modulation_index` e perfil por bins de fase.

## Fluxo
1. Discretizar fase lenta em bins.
2. Acumular amplitude rápida por bin.
3. Estimar MI normalizado e publicar no payload auxiliar.

## Custo computacional
O(T + B) por janela, com B bins de fase; memória pequena para histogramas.

## Integração
- `compute_phase_amplitude_coupling` (`src/pyfolds/advanced/speech_tracking.py`).
- `WaveDynamicsMixin.forward` injeta resultado em `extra_payload` (`src/pyfolds/advanced/wave.py`).
- `MPJRDConfig.enable_cross_frequency_coupling` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** estimativa em lote único e sem rotina de surrogates/controle estatístico.
