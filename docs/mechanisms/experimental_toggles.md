# Experimental Toggles

## Objetivo
Centralizar ativação de mecanismos experimentais para execução controlada de ablações.

## Variáveis
- **Entrada:** objeto de configuração (`cfg`).
- **Controle:** flags `enable_*` e `experimental_*`.
- **Saída:** conjunto de toggles e especificações de mecanismo.

## Fluxo
1. Ler flags do `MPJRDConfig`.
2. Materializar estrutura de toggles ativa/desativada.
3. Expor metadados para comparação baseline vs experimento.

## Custo computacional
O(M) no número de mecanismos registrados; custo irrelevante frente ao forward.

## Integração
- `ExperimentalMechanismConfig.from_config` (`src/pyfolds/advanced/experimental.py`).
- `MechanismToggleSet` e `MechanismSpec` (`src/pyfolds/advanced/experimental.py`).
- Flags em `MPJRDConfig` (`src/pyfolds/core/config.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** camada de controle é simples, determinística e usada para governar features de experimento.
