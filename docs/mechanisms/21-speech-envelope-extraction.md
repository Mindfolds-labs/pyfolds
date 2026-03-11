# Speech Envelope Extraction

## Objetivo
Extrair envelope acústico da fala para alimentar mecanismos oscilatórios e métricas de sincronização neural.

## Variáveis
- **Entrada:** `audio`, `sample_rate`, `method` (`hilbert` ou `gammatone`).
- **Controle:** `enable_speech_envelope_tracking`, `speech_envelope_method`.
- **Saída:** `envelope`, `onset_strength`, `modulation_spectrum`.

## Fluxo
1. Converter o sinal para 1D e normalizar formato de tensor.
2. Calcular envelope por `_analytic_signal_hilbert` ou `_gammatone_envelope`.
3. Derivar onsets e espectro de modulação para uso pela dinâmica de onda.

## Custo computacional
Predominantemente O(T log T) no caminho Hilbert (FFT) e O(B·T) no caminho gammatone aproximado; memória linear em T.

## Integração
- `extract_speech_envelope` (`src/pyfolds/advanced/speech_tracking.py`).
- `_analytic_signal_hilbert` e `_gammatone_envelope` (`src/pyfolds/advanced/speech_tracking.py`).
- `WaveDynamicsMixin.forward` consome o envelope (`src/pyfolds/advanced/wave.py`).
- `MPJRDConfig.enable_speech_envelope_tracking` e `MPJRDConfig.speech_envelope_method` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** o método gammatone é aproximado e ainda sem validação extensiva em corpus externo.
