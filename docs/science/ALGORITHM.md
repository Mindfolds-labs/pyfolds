# ALGORITHM — Processamento Dendrítico Assimétrico (formato científico)

> Objetivo: tornar o modelo matemático legível no GitHub Markdown, com notação estável para documentação técnica e futura publicação.

## 1) Formulação matemática canônica

A dinâmica do neurônio MPJRD no runtime é definida por integração dendrítica local, integração/competição configurável entre ramos e decisão somática.

\[
\boxed{\;y = H(u-\theta)\;}
\]

onde \(H(\cdot)\) é a função degrau de Heaviside e

\[
u = f_{int}(v,\theta;m),
\qquad
v_d = \sigma_d\!\left(\sum_{s=1}^{S} \psi\big(N_{d,s},W_{d,s},I_{d,s}\big)\,x_{d,s}\right).
\]

### Integração entre ramos (runtime atual)

O runtime atual não usa WTA duro como mecanismo único. O comportamento depende de `dendrite_integration_mode`:

- `nmda_shunting`: gate sigmoidal por dendrito + normalização divisiva (`DendriticIntegration`);
- `wta_soft`: soma cooperativa de gates sigmoidais;
- `wta_hard`: seleção vencedora (mantida para compatibilidade/ablação).

Forma genérica:

\[
\mathbf{u} = f_{int}(\mathbf{v}, \theta; m),\quad m\in\{\texttt{nmda\_shunting},\texttt{wta\_soft},\texttt{wta\_hard}\}.
\]

## 2) Tabela de símbolos (padrão de documentação técnica)

| Símbolo | Significado | Dimensão típica |
|---|---|---|
| \(x_{d,s}\) | entrada no ramo \(d\), sinapse \(s\) | \(\mathbb{R}\) |
| \(N_{d,s}\) | traço/plasticidade local | \(\mathbb{R}\) |
| \(W_{d,s}\) | peso sináptico | \(\mathbb{R}\) |
| \(I_{d,s}\) | estado interno/inibição local | \(\mathbb{R}\) |
| \(v_d\) | potencial dendrítico após não linearidade | \(\mathbb{R}\) |
| \(g_d\) | gate/ganho por ramo (binário ou contínuo conforme modo) | \(\mathbb{R}_{\ge 0}\) |
| \(u\) | potencial somático agregado | \(\mathbb{R}\) |
| \(\theta\) | limiar de disparo | \(\mathbb{R}\) |
| \(y\) | spike de saída | \(\{0,1\}\) |

## 3) Mapeamento direto para implementação

No `forward` de `MPJRDNeuron`, o fluxo computacional é:

1. **Integração local por ramo**
   \[
   v_d \leftarrow \texttt{dend}(x[:,d,:])
   \]
2. **Integração entre ramos (por modo)**
   \[
   u \leftarrow f_{int}(v,\theta; m)
   \]
3. **Disparo**
   \[
   y \leftarrow \mathbb{1}[u\ge\theta_{eff}]
   \]
4. **Homeostase, neuromodulação, acumulação batch e telemetria**.

## 4) Pseudocódigo em estilo científico (IEEE-friendly)

```text
Algorithm 1 MPJRD Forward and Deferred Plasticity
Input: X ∈ R^{B×D×S}, estados {N,W,I}, θ, mode
Output: Y ∈ {0,1}^B, estados intermediários e telemetria

for d = 1..D do
    V[:, d] ← DENDRITE_INTEGRATE(X[:, d, :], N_d, W_d, I_d)
end for

U ← INTEGRATE_DENDRITES(V, θ, mode)
Y ← 1[U ≥ θ_eff]

if mode ≠ INFERENCE then
    UPDATE_HOMEOSTASIS(Y)
end if

R ← COMPUTE_NEUROMODULATION(signal_ext, state)

if mode = BATCH and defer_updates = true then
    ACCUMULATE_BATCH_STATS(X, U, Y, R)
end if

EMIT_TELEMETRY(U, Y, R)
return Y
```

## 5) Regra consolidada de plasticidade (batch)

No `apply_plasticity`, com estatísticas acumuladas:

\[
\bar{x}_{d,s} = \frac{1}{T}\sum_{t=1}^{T}x_{d,s}^{(t)},
\qquad
\bar{u}_d = \frac{1}{T}\sum_{t=1}^{T}u_d^{(t)},
\qquad
\rho_{\text{post}} = \frac{1}{T}\sum_{t=1}^{T}y^{(t)}.
\]

A taxa pré efetiva por dendrito é modulada por atividade antes de chamar `update_synapses_rate_based(..., mode=self.mode)`.

## 6) Invariante de correção (anti-degeneração)

Para não colapsar para um perceptron quase linear, deve-se preservar:

\[
\text{(não linearidade local por ramo)} \Rightarrow \text{(integração/competição)} \Rightarrow \text{(agregação somática)}.
\]

Violação típica (incorreta): somar todos os sinais sinápticos antes da transformação local.

## 7) Complexidade assintótica

- Integração dendrítica: \(\mathcal{O}(BDS)\)
- Integração/competição entre ramos por batch: \(\mathcal{O}(BD)\)
- Estado principal (`N`, `W`, `I`): \(\mathcal{O}(DS)\)

## 8) Guia rápido para manter renderização limpa no GitHub

- Use blocos `\[ ... \]` para equações principais e `\( ... \)` para inline.
- Evite macros LaTeX avançadas não suportadas pelo renderizador do GitHub.
- Sempre manter **Tabela de Símbolos** no topo de documentos matemáticos.
- Para publicação futura (PDF/IEEE), mantenha numeração de algoritmo e nomenclatura estáveis.

## 9) Fluxo real do neurônio

Referências diretas de implementação: `MPJRDNeuron` (`src/pyfolds/core/neuron.py`), `MPJRDNeuronV2` (`src/pyfolds/core/neuron_v2.py`), `DendriticIntegration` (`src/pyfolds/core/dendrite_integration.py`), `AdaptationMixin` (`src/pyfolds/advanced/adaptation.py`) e `RefractoryMixin` (`src/pyfolds/advanced/refractory.py`).

O passo efetivo é: integração dendrítica → integração entre ramos por modo → decisão por `theta_eff` → homeostase (quando aplicável) → plasticidade (`_apply_online_plasticity`/`apply_plasticity`) → telemetria.

Ver também [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) e [`docs/mechanisms/experimental_toggles.md`](../mechanisms/experimental_toggles.md).
