# Envelope Event Detection

## Objetivo
Detectar bordas acústicas relevantes no envelope para disparar mecanismos dependentes de eventos.

## Variáveis
- **Entrada:** `envelope` temporal.
- **Controle:** limiar adaptativo interno do detector.
- **Saída:** `onset_times`, `onset_strength`, `event_mask`.

## Fluxo
1. Calcular derivada temporal do envelope.
2. Retificar valores positivos como candidatos a onset.
3. Aplicar threshold adaptativo e emitir máscara/eventos.

## Custo computacional
O(T) em tempo e memória linear; custo baixo frente ao restante da simulação.

## Integração
- `detect_envelope_events` (`src/pyfolds/advanced/speech_tracking.py`).
- `WaveDynamicsMixin.forward` usa os eventos para reset/gating (`src/pyfolds/advanced/wave.py`).
- `analyze_mechanisms` agrega sinais de eventos para diagnóstico (`src/pyfolds/advanced/speech_tracking.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** estratégia de limiar único pode degradar em cenários de SNR baixo.
