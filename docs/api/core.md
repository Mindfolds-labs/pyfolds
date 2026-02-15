# API — Core

## Classes principais

| Classe | Módulo | Função |
|---|---|---|
| `MPJRDConfig` | `pyfolds.core.config` | Hiperparâmetros e validação |
| `MPJRDNeuron` | `pyfolds.core.neuron` | Pipeline completo de forward/plasticidade |
| `MPJRDDendrite` | `pyfolds.core.dendrite` | Integração de sinapses por ramo |
| `MPJRDSynapse` | `pyfolds.core.synapse` | Estado sináptico (`N`, `I`, `W`) |
| `HomeostasisController` | `pyfolds.core.homeostasis` | Controle de `theta` e `r_hat` |
| `Neuromodulator` | `pyfolds.core.neuromodulation` | Cálculo de sinal `R` |
| `StatisticsAccumulator` | `pyfolds.core.accumulator` | Acúmulo de estatísticas para batch/sleep |

## Fluxo do `MPJRDNeuron.forward`

1. Integração dendrítica (`v_dend`).
2. Gate competitivo por dendrito (WTA no estado atual).
3. Potencial somático `u`.
4. Disparo (`spikes`) via `theta`.
5. Homeostase, neuromodulação e acúmulo opcional.

## Factories úteis

- `pyfolds.core.create_neuron(...)`
- `pyfolds.core.create_accumulator(cfg, track_extra=False)`
