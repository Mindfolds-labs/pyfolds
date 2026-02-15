# Scientific Basis — MPJRD-Wave v3.0

## Objetivo científico
A versão MPJRD-Wave (v3.0) mantém o núcleo de plasticidade estrutural da v2.0 e adiciona codificação temporal por fase/frequência para ampliar a capacidade computacional sem multiplicar profundidade de rede.

## Mapeamento literatura → componente

### 1) Bartol et al. (2015): quantização estrutural e escala logarítmica
- Evidência: força sináptica correlacionada a estados estruturais discretos de espinhas.
- Tradução no PyFolds:
  - Mantém-se `N` como estado estrutural discreto.
  - Peso sináptico continua `W = log2(1 + N) / w_scale`.
  - **Novo na v3.0:** amplitude axonal usa escala compatível: `A = log2(1 + U_somático)`.

### 2) Poirazi & Mel (2001): neurônio como rede de duas camadas
- Evidência: dendritos como subunidades não-lineares elevam poder de representação.
- Tradução no PyFolds:
  - Cada dendrito processa soma local e aplica não-linearidade (`sigmoid`).
  - Soma recebe integrações já não-lineares (cooperação), sem WTA rígido.

### 3) Gidon et al. (2020): spikes dendríticos locais
- Evidência: ramos dendríticos executam computação local e não somente condução passiva.
- Tradução no PyFolds:
  - `dendritic_activations = sigmoid(v_dend - threshold)` representa disparo local de ramo.
  - O potencial somático passa a refletir sinergia entre múltiplos ramos.

### 4) Thorpe et al. (2001) + Hopfield (1995): tempo/fase como código
- Evidência: latência e ordem temporal carregam informação.
- Tradução no PyFolds:
  - Fase derivada de `U` com regra monotônica inversa (maior U → menor atraso de fase).
  - Saída em quadratura (real/imag) preserva informação completa de fase.
  - Frequência representa “canal/categoria”; fase representa confiança e prioridade temporal.

## Hipótese operacional de v3.0
A precisão de classificação emerge da combinação:
1. Memória estrutural (`N`) estável e plástica.
2. Integração cooperativa dendrítica (não-linear).
3. Competição temporal por fase (quem entra em fase cedo domina leitura populacional).

## Referências-chave
- Bartol, T. M. et al. (2015). *Nanoconnectomic upper bound on the variability of synaptic plasticity.* eLife.
- Poirazi, P., Brannon, T., & Mel, B. W. (2003/2001 linha de trabalho). *Pyramidal neuron as two-layer neural network.*
- Gidon, A. et al. (2020). *Dendritic action potentials and computation in human layer 2/3 cortical neurons.* Science.
- Thorpe, S., Delorme, A., & Van Rullen, R. (2001). *Spike-based strategies for rapid processing.*
- Hopfield, J. J. (1995). *Pattern recognition computation using action potential timing.*
