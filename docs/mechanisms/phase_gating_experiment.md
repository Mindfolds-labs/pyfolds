# Phase Gating Experiment

## Objetivo
Aplicar modulação dependente de fase no update STDP para testar aprendizado seletivo em janelas favoráveis.

## Variáveis
- **Entrada:** `base_delta_w`, `phase`.
- **Controle:** `enable_phase_gating`.
- **Saída:** `delta_w` modulado e métricas de aprendizado associadas.

## Fluxo
1. Calcular update STDP base.
2. Se toggle ativo, aplicar fator `max(0, cos(phase))`.
3. Registrar impacto em métricas de atividade/plasticidade.

## Custo computacional
O(1) adicional por atualização sináptica (cosseno + clamp), sem aumento significativo de memória.

## Integração
- `STDPMixin._update_stdp_traces` (`src/pyfolds/advanced/stdp.py`).
- `MechanismToggleSet.is_enabled` com chave `phase_gating` (`src/pyfolds/advanced/experimental.py`).
- `MPJRDConfig.enable_phase_gating` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Experimental`.
- **Justificativa:** técnica está explicitamente guardada por toggle e sem validação final de generalização.
