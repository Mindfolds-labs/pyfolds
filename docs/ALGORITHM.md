# ALGORITHM — Processamento Dendrítico Assimétrico

## 1) Equação operacional

A dinâmica pode ser lida como:

\[
\text{Output} = \phi\left(\sum_{d=1}^{D} g_d\,v_d\right)
\]

com:

\[
v_d = \sigma_d\left(\sum_{s=1}^{S} f(N_{d,s},W_{d,s},I_{d,s})\cdot x_{d,s}\right), \quad g_d \in \{0,1\}
\]

No modo atual do código, `g_d` é definido por **Winner-Take-All** (somente o dendrito vencedor ativa, `\sum_d g_d = 1`).

## 2) Mapeamento direto para o código

No `forward` de `MPJRDNeuron`:
1. `v_dend[:, d] = dend(x[:, d, :])`  
   → integração e não linearidade local por ramo.
2. `max` + `scatter`  
   → gating WTA.
3. `u = gated.sum(dim=1)`  
   → potencial somático.
4. `spikes = (u >= theta).float()`  
   → disparo homeostático.

## 3) Pseudocódigo científico

```text
Entrada: x[B, D, S], estado sináptico {N,W,I}, limiar θ
Saída: spikes[B], potenciais internos e estatísticas

1. Para cada d em [1..D]:
      v_d[d] ← DENDRITE_INTEGRATE(x[:, d, :], N_d, W_d, I_d)

2. k ← argmax_d(v_d)
   gated ← 0
   gated[k] ← v_d[k]

3. u ← soma_d(gated[d])
4. spikes ← 1[u ≥ θ]

5. Atualizar homeostase (se modo != INFERENCE)
6. Calcular neuromodulação R (externa ou endógena)
7. Se modo BATCH e defer_updates:
      acumular (x, gated, spikes)

8. Emitir telemetria + retorno estruturado
```

## 4) Regra de plasticidade consolidada (batch)

Em `apply_plasticity`:
- Recupera médias acumuladas (`x_mean`, `gated_mean`, `post_rate`).
- Deriva taxa pré por dendrito com máscara de atividade.
- Atualiza sinapses por `dend.update_synapses_rate_based(..., mode=self.mode)`.

## 5) Verificação do ponto crítico (debug científico)

**Falha comum:** somar sinais de todos os ramos antes da não linearidade local, colapsando para comportamento quase linear.

**Estado atual:** o código já aplica transformação por dendrito antes da agregação somática e usa competição WTA, preservando compartimentalização funcional.

## 6) Complexidade

- Integração dendrítica: \(O(B\cdot D\cdot S)\)
- WTA por batch: \(O(B\cdot D)\)
- Memória principal: estados `N, W, I` em \(O(D\cdot S)\)
