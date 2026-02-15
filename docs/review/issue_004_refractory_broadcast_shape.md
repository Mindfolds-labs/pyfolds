# Issue 004 — Inconsistência dimensional no refratário causa erro em cascata na adaptação

**Tipo:** bug matemático/implementação  
**Severidade:** alta  
**Status:** corrigido

## Contexto
No `RefractoryMixin.forward`, o termo de limiar efetivo era computado como:

```python
theta_eff = output['theta'] + theta_boost.unsqueeze(1)
```

Com `B` amostras no batch:
- `output['u']` tem forma `[B]`
- `theta_boost.unsqueeze(1)` tem forma `[B, 1]`

Na comparação `output['u'] >= theta_eff`, o PyTorch faz broadcasting para `[B, B]`, quebrando a semântica por amostra do modelo refratário.

## Impacto matemático
A dinâmica refratária é definida por amostra, isto é, para cada neurônio/instância `b`:

\[
spike_b = \mathbb{1}\{u_b \ge \theta_b\}
\]

A expansão indevida para `[B, B]` mistura comparações cruzadas entre amostras, violando a independência estatística do batch e propagando erro para mixins subsequentes (ex.: `AdaptationMixin` espera `spikes` em `[B]`).

## Correção aplicada
Substituição para manter `theta_eff` em `[B]`:

```python
theta_eff = output['theta'] + theta_boost
```

Assim, a operação de limiar volta a ser ponto-a-ponto e consistente com a formulação do refratário por amostra.

## Validação
- `tests/integration/test_neuron_advanced.py::TestAdvancedNeuron::test_full_pipeline` passa.
- `tests/unit/advanced/test_refractory.py` passa.

