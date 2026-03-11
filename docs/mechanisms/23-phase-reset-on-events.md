# Phase Reset on Events

## Objetivo
Realinhar fase oscilatória após eventos acústicos fortes para reduzir drift e melhorar locking temporal.

## Variáveis
- **Entrada:** `phase`, `event_strength`.
- **Controle:** `enable_phase_reset_on_audio_event`, `phase_reset_threshold`, `phase_reset_target`.
- **Saída:** fase corrigida usada na dinâmica subsequente.

## Fluxo
1. Receber força do evento detectado no envelope.
2. Comparar força com threshold configurado.
3. Substituir/interpolar fase para o alvo quando o critério é satisfeito.

## Custo computacional
O(1) por passo para decisão de reset; impacto de memória desprezível.

## Integração
- `reset_phase_if_event` (`src/pyfolds/advanced/speech_tracking.py`).
- `WaveDynamicsMixin.forward` aplica reset condicional (`src/pyfolds/advanced/wave.py`).
- Flags em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** ainda opera com reset global simplificado, sem granularidade espacial fina.
