# Relatório de Erros Corrigidos - PyFolds

Data: 17/02/2026

Este relatório consolida os erros lógicos/semânticos identificados na análise de código e o status de correção aplicado no código-fonte.

## Erros encontrados

1. **Broadcasting incorreto em `refractory.py`** (ALTO)
   - Risco: `theta` escalar ou shape `[1]` combinado com `theta_boost` `[B]` poderia gerar shape inesperado.
   - Correção: normalização explícita de `theta` para o batch e validação de shape incompatível.

2. **Campos de output não garantidos em `adaptation.py`** (MÉDIO)
   - Risco: `KeyError` ao acessar `output['u']` e `output['spikes']` sem validação.
   - Correção: verificação explícita dos campos obrigatórios antes de aplicar adaptação.

3. **Shape inconsistente em `inhibition.py`** (ALTO)
   - Risco: mismatch entre número de neurônios excitadores esperados e shape recebido em `exc_output['spikes']`.
   - Correção: validações de shape de `exc_spikes` e `inh_spikes` com erro claro.

4. **Estado potencialmente não inicializado em `backprop.py`** (MÉDIO)
   - Risco: uso de `backprop_trace` sem garantir alocação para o batch atual.
   - Correção: chamada de `_ensure_backprop_trace` antes do decaimento da trace durante processamento da fila.

5. **Falta de validação em `short_term.py`** (MÉDIO)
   - Risco: entrada com dimensões incorretas gerava comportamento silencioso incorreto.
   - Correção: validação de dimensionalidade `[B, D, S]` e compatibilidade com config.

6. **Ambiguidade no fallback de `theta/thetas` em `inhibition.py`** (MÉDIO)
   - Risco: fallback silencioso para threshold `0` mascarava erro de integração.
   - Correção: exigência explícita de `theta`/`thetas` e validação robusta de shape antes da comparação.

7. **Possível NaN em `neuromodulation.py`** (MÉDIO)
   - Risco: `saturation_ratio`, `rate_error` ou `R_val` podiam propagar NaN.
   - Correção: validação de NaN, clamp do intervalo e erros explícitos no modo `capacity`.

## Status

- ✅ Erros de severidade **ALTA** corrigidos (#1 e #3).
- ✅ Erros de severidade **MÉDIA** corrigidos (#2, #4, #5, #6, #7).

