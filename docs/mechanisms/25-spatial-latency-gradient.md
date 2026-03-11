# Spatial Latency Gradient

## Objetivo
Introduzir atraso dependente da posição neural para modelar propagação espaço-temporal.

## Variáveis
- **Entrada:** coordenadas neurais e latência base.
- **Controle:** `enable_spatial_latency_gradient`, `spatial_latency_max_ms`, `spatial_latency_scale`.
- **Saída:** `spatial_latency_ms` aplicado no passo de onda.

## Fluxo
1. Ler coordenada/posição do neurônio.
2. Calcular atraso com `latency_kernel` e saturação.
3. Somar atraso à latência total utilizada pela dinâmica.

## Custo computacional
O(N) sobre número de neurônios avaliados; memória linear no vetor de latências.

## Integração
- `latency_kernel` (`src/pyfolds/advanced/speech_tracking.py`).
- `WaveDynamicsMixin.forward` aplica latência espacial (`src/pyfolds/advanced/wave.py`).
- Parâmetros em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** kernel isotrópico simples ainda não incorpora conectividade anatômica real.
