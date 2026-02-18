# ALGORITHM — Processamento Dendrítico Assimétrico (formato científico)

> Objetivo: tornar o modelo matemático legível no GitHub Markdown, com notação estável para documentação técnica e futura publicação.

## 1) Formulação matemática canônica

A dinâmica do neurônio MPJRD é definida por três estágios: integração dendrítica, competição espacial e decisão somática.

\[
\boxed{\;y = H(u-\theta)\;}
\]

onde \(H(\cdot)\) é a função degrau de Heaviside e

\[
u = \sum_{d=1}^{D} g_d\,v_d,
\qquad
v_d = \sigma_d\!\left(\sum_{s=1}^{S} \psi\big(N_{d,s},W_{d,s},I_{d,s}\big)\,x_{d,s}\right).
\]

### Gating por Winner-Take-All (WTA)

No modo atual:

\[
g_d = \mathbb{1}\!\left[d = \arg\max_{j\in\{1,\dots,D\}} v_j\right],
\qquad
\sum_{d=1}^{D} g_d = 1.
\]

Isso garante seletividade espacial e evita a soma indiscriminada de ramos.

## 2) Tabela de símbolos (padrão de documentação técnica)

| Símbolo | Significado | Dimensão típica |
|---|---|---|
| \(x_{d,s}\) | entrada no ramo \(d\), sinapse \(s\) | \(\mathbb{R}\) |
| \(N_{d,s}\) | traço/plasticidade local | \(\mathbb{R}\) |
| \(W_{d,s}\) | peso sináptico | \(\mathbb{R}\) |
| \(I_{d,s}\) | estado interno/inibição local | \(\mathbb{R}\) |
| \(v_d\) | potencial dendrítico após não linearidade | \(\mathbb{R}\) |
| \(g_d\) | gate competitivo do ramo | \(\{0,1\}\) |
| \(u\) | potencial somático agregado | \(\mathbb{R}\) |
| \(\theta\) | limiar de disparo | \(\mathbb{R}\) |
| \(y\) | spike de saída | \(\{0,1\}\) |

## 3) Mapeamento direto para implementação

No `forward` de `MPJRDNeuron`, o fluxo computacional é:

1. **Integração local por ramo**
   \[
   v_d \leftarrow \texttt{dend}(x[:,d,:])
   \]
2. **Competição WTA**
   \[
   k \leftarrow \arg\max_d(v_d), \quad g_k=1,\; g_{d\neq k}=0
   \]
3. **Agregação somática**
   \[
   u \leftarrow \sum_d g_d v_d
   \]
4. **Disparo**
   \[
   y \leftarrow \mathbb{1}[u\ge\theta]
   \]
5. **Homeostase, neuromodulação, acumulação batch e telemetria**.

## 4) Pseudocódigo em estilo científico (IEEE-friendly)

```text
Algorithm 1 MPJRD Forward and Deferred Plasticity
Input: X ∈ R^{B×D×S}, estados {N,W,I}, θ, mode
Output: Y ∈ {0,1}^B, estados intermediários e telemetria

for d = 1..D do
    V[:, d] ← DENDRITE_INTEGRATE(X[:, d, :], N_d, W_d, I_d)
end for

K ← argmax_d V[:, d]
G ← one_hot(K, D)
U ← sum_d (G[:, d] ⊙ V[:, d])
Y ← 1[U ≥ θ]

if mode ≠ INFERENCE then
    UPDATE_HOMEOSTASIS(Y)
end if

R ← COMPUTE_NEUROMODULATION(signal_ext, state)

if mode = BATCH and defer_updates = true then
    ACCUMULATE_BATCH_STATS(X, G, Y, R)
end if

EMIT_TELEMETRY(U, Y, R)
return Y
```

## 5) Regra consolidada de plasticidade (batch)

No `apply_plasticity`, com estatísticas acumuladas:

\[
\bar{x}_{d,s} = \frac{1}{T}\sum_{t=1}^{T}x_{d,s}^{(t)},
\qquad
\bar{g}_d = \frac{1}{T}\sum_{t=1}^{T}g_d^{(t)},
\qquad
\rho_{\text{post}} = \frac{1}{T}\sum_{t=1}^{T}y^{(t)}.
\]

A taxa pré efetiva por dendrito é modulada por atividade (
\(\bar{g}_d\)) antes de chamar `update_synapses_rate_based(..., mode=self.mode)`.

## 6) Invariante de correção (anti-degeneração)

Para não colapsar para um perceptron quase linear, deve-se preservar:

\[
\text{(não linearidade local por ramo)} \Rightarrow \text{(competição)} \Rightarrow \text{(agregação somática)}.
\]

Violação típica (incorreta): somar todos os sinais sinápticos antes da transformação local.

## 7) Complexidade assintótica

- Integração dendrítica: \(\mathcal{O}(BDS)\)
- WTA por batch: \(\mathcal{O}(BD)\)
- Estado principal (`N`, `W`, `I`): \(\mathcal{O}(DS)\)

## 8) Guia rápido para manter renderização limpa no GitHub

- Use blocos `\[ ... \]` para equações principais e `\( ... \)` para inline.
- Evite macros LaTeX avançadas não suportadas pelo renderizador do GitHub.
- Sempre manter **Tabela de Símbolos** no topo de documentos matemáticos.
- Para publicação futura (PDF/IEEE), mantenha numeração de algoritmo e nomenclatura estáveis.
