# ISSUE-003 — Inconsistência matemática entre ONLINE e BATCH no termo pré-sináptico

## Contexto
Durante auditoria do core (`MPJRDNeuron.apply_plasticity`), foi identificado que o cálculo de `pre_rate` em modo **BATCH** fazia:

```python
active_mask = (x_mean[d_idx] > activity_threshold).float()
n_active = active_mask.sum().clamp_min(1.0)
pre_rate = (x_mean[d_idx] * active_mask) / n_active
```

`x_mean[d_idx]` já representa a média por sinapse \(\mathbb{E}[x_j]\) acumulada no batch.

## Problema técnico
A divisão adicional por `n_active` cria um fator global acoplado ao número de sinapses ativas do dendrito, alterando o valor local de cada sinapse:

\[
\tilde{x}_j = \frac{\mathbb{E}[x_j] \cdot \mathbb{1}[x_j > \tau]}{\sum_k \mathbb{1}[x_k > \tau]}
\]

Isso viola a localidade da regra e quebra a equivalência esperada entre atualização ONLINE (média temporal por sinapse) e BATCH (acumulada e depois aplicada).

## Impacto observado/esperado
- **Subescala sistemática** de `pre_rate` quando múltiplas sinapses ficam ativas.
- **Aprendizado mais lento** e dependente da dimensionalidade do dendrito (efeito espúrio).
- Potencial distorção de estabilidade/convergência ao alterar o balanço LTP/LTD.

## Correção aplicada
Foi removida a normalização cruzada por `n_active` no caminho BATCH e mantido apenas o mascaramento por atividade:

```python
pre_rate = x_mean[d_idx] * active_mask
pre_rate = pre_rate.clamp(0.0, 1.0)
```

## Fundamentação científica (resumo)
Regras de plasticidade baseadas em taxa do tipo Hebb/Oja usam termos locais por sinapse (pré x pós), sem normalização transversal dependente de quantas outras sinapses estão ativas no mesmo instante.

Referências clássicas:
1. Hebb, D. O. (1949). *The Organization of Behavior*.
2. Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*.
3. Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal Dynamics*.

## Evidência de validação
Foi adicionado teste unitário cobrindo o cenário: `test_batch_plasticity_preserves_local_pre_synaptic_rate`.

