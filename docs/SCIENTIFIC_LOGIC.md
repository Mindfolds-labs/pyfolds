# Scientific Logic — PyFolds (MPJRD v2.0 / MPJRD-Wave v3.0)

## Objetivo

Este documento descreve a lógica científica que fundamenta o PyFolds, conectando cada decisão de arquitetura do framework com evidências da literatura neurocientífica e neurocomputacional.

---

## 1. Pilares científicos do modelo

### 1.1 Quantização sináptica e memória estrutural (Bartol et al., 2015)

Bartol et al. (2015) mostraram que sinapses corticais apresentam **resolução finita de estados estruturais**, com capacidade estimada próxima de ~4.7 bits por sinapse, equivalente a ~26 níveis discerníveis. O PyFolds traduz esse princípio para um modelo operacional usando `N` (filamentos) no intervalo padrão `0..31` (32 estados possíveis; 31 transições úteis), garantindo memória discreta auditável.

No MPJRD:

- `N` representa o estado estrutural de longo prazo.
- `I` representa o potencial interno de curto prazo.
- `W` é derivado de `N` por mapeamento logarítmico:

\[
W(N) = \frac{\log_2(1 + N)}{w_{scale}}
\]

Consequência prática: pequenas flutuações em atividade não mudam diretamente a memória estrutural; primeiro elas acumulam em `I`, e só depois promovem/rebaixam `N` via limiares (`i_ltp_th`, `i_ltd_th`). Isso reduz instabilidade e aumenta interpretabilidade causal.

---

### 1.2 Subunidades dendríticas e neurônio de duas camadas (Poirazi & Mel, 2001)

Poirazi & Mel (2001) propõem que neurônios piramidais podem ser aproximados como uma rede de **duas camadas**:

1. camada de subunidades dendríticas não lineares;
2. camada somática de integração/decisão.

O PyFolds implementa diretamente essa hipótese:

- cada `MPJRDDendrite` integra sinapses localmente com não linearidade;
- `MPJRDNeuron` agrega saídas dendríticas (`v_dend`) para formar `u`;
- o spike (`spikes`) é decidido no soma usando limiar adaptativo `theta`.

Isso justifica arquiteturalmente por que a não-linearidade deve acontecer **antes** da combinação global.

---

### 1.3 Spikes dendríticos e computação local (Gidon et al., 2020)

Gidon et al. (2020) mostram evidências de que dendritos humanos geram eventos ativos (incluindo spikes dendríticos), ampliando poder computacional local e sensibilidade a padrões cooperativos.

No PyFolds, essa ideia aparece em dois níveis:

- **v2.0 (MPJRD):** competição/gating entre ramos após computação local;
- **v3.0 (MPJRD-Wave):** integração cooperativa contínua entre ramos via ativação sigmoid por dendrito (`dendritic_activations`), reduzindo perda de informação causada por seleção dura.

Interpretação: o dendrito deixa de ser “fio passivo” e passa a ser unidade funcional que participa de inferência, plasticidade e explicação.

---

### 1.4 Codificação por fase e latência (O'Keefe & Recce, 1993; Thorpe et al., 2001)

A arquitetura MPJRD-Wave v3.0 é inspirada em dois eixos experimentais/computacionais:

- **O'Keefe & Recce (1993):** informação pode ser codificada por fase relativa em oscilações neurais;
- **Thorpe et al. (2001):** códigos temporais/latência suportam reconhecimento rápido e eficiente.

No MPJRD-Wave:

- amplitude funcional: `amplitude = log2(1 + relu(u))`;
- fase dinâmica: `phase = (π/2) * (1 - sigmoid((u-theta)*phase_sensitivity))`;
- frequência por classe/contexto: `base_frequency + k*frequency_step` ou `class_frequencies`;
- saída em quadratura: `wave_real`, `wave_imag`, `wave_complex`.

Assim, a saída não codifica apenas “disparou/não disparou”, mas também **quando** e **em qual fase/frequência** a evidência foi expressa.

---

## 2. Mapeamento científico → componentes do framework

| Conceito científico | Implementação no PyFolds | Variáveis observáveis |
|---|---|---|
| Quantização estrutural | `MPJRDSynapse` | `N`, `I`, `W`, `protection`, `sat_time` |
| Subunidade dendrítica | `MPJRDDendrite` | `v_dend`, `dendritic_activations` |
| Integração somática adaptativa | `MPJRDNeuron` + `HomeostasisController` | `u`, `theta`, `r_hat`, `spikes` |
| Modulação contextual | `Neuromodulator` | `R` |
| Código temporal/fase | `MPJRDWaveNeuron` | `phase`, `latency`, `frequency`, `wave_real/imag` |

---

## 3. Hipótese operacional do PyFolds

A hipótese central do framework é:

> Modelos com memória estrutural discreta (`N`), subunidades dendríticas explícitas e controle homeostático/modulatório produzem comportamento mais auditável e biologicamente plausível do que MLPs com pesos contínuos opacos.

### Predições práticas esperadas

1. Melhor rastreabilidade causal de aprendizado (transições de `N`).
2. Menor sensibilidade a ruído de curto prazo (filtragem via `I`).
3. Maior separabilidade de padrões compostos (dendritos como especialistas locais).
4. Em v3.0, maior expressividade temporal por fase/frequência.

---

## 4. Referências

- Bartol, T. M. et al. (2015). *Nanoconnectomic upper bound on the variability of synaptic plasticity*. eLife.
- Poirazi, P., & Mel, B. W. (2001). *Impact of active dendrites and structural plasticity on the memory capacity of neural tissue*. Neuron.
- Gidon, A. et al. (2020). *Dendritic action potentials and computation in human layer 2/3 cortical neurons*. Science.
- O'Keefe, J., & Recce, M. L. (1993). *Phase relationship between hippocampal place units and the EEG theta rhythm*. Hippocampus.
- Thorpe, S., Delorme, A., & Van Rullen, R. (2001). *Spike-based strategies for rapid processing*. Neural Networks.
