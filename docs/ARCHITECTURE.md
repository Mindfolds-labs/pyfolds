# ARCHITECTURE — PyFolds (C4 + Runtime Sequence)

Este documento descreve a arquitetura do PyFolds usando a abordagem C4 e um diagrama de sequência do `forward`.

## 1. C4 — Contexto (C1)

```mermaid
flowchart LR
    U[Pesquisador / Engenheiro de ML]
    P[PyFolds\nMPJRD v2 + MPJRD-Wave v3]
    T[PyTorch\nTensores / Autograd / Device]

    U -->|Configura experimento, treina, analisa métricas| P
    P -->|Executa operações tensorizadas| T
    T -->|Backend numérico (CPU/GPU)| P
```

**Escopo do sistema:** modelagem neural bioinspirada com sinapse estrutural, processamento dendrítico, homeostase, neuromodulação e extensões de fase/frequência.

---

## 2. C4 — Containers (C2)

```mermaid
flowchart TB
    subgraph PyFolds
      C1[core\nConfig, Synapse, Dendrite, Neuron, Homeostasis, Neuromodulation]
      C2[advanced\nSTDP, adaptation, inhibition, refractory, short-term, backprop]
      C3[wave\nWaveConfig, WaveNeuron\nIntegração cooperativa + fase/frequência]
      C4[utils\nTipos, matemática, logging, helpers]
    end

    User[Usuário/Pesquisador] --> C1
    User --> C2
    User --> C3
    C1 --> C4
    C2 --> C1
    C2 --> C4
    C3 --> C1
    C3 --> C4
```

---

## 3. C4 — Componentes do módulo `core` (C3)

```mermaid
classDiagram
    class MPJRDNeuron {
      +forward(x, reward, mode, collect_stats, dt)
      +apply_plasticity(dt, reward)
      +sleep(duration)
      +get_metrics()
      -_compute_R_endogenous(current_rate, saturation_ratio)
      -_apply_online_plasticity(x, post_rate, R_tensor, dt, mode)
    }

    class MPJRDDendrite {
      +forward(x)
      +update_synapses_rate_based(pre_rate, post_rate, R, dt, mode)
      +consolidate(dt)
      +sleep_step(dt)
      +N_mean
      +I_mean
      +W_mean
    }

    class MPJRDSynapse {
      +N: int
      +I: float
      +W: float
      +protection: bool
      +sat_time: float
      +update(pre_rate, post_rate, R, dt, mode)
      +consolidate(dt)
      +sleep_step(dt)
      +reset()
    }

    class HomeostasisController {
      +update(spike_rate)
      +theta
      +r_hat
    }

    class Neuromodulator {
      +forward(external_reward, current_rate, saturation_ratio, r_hat)
      +mode
    }

    MPJRDNeuron --> MPJRDDendrite : compõe D dendritos
    MPJRDDendrite --> MPJRDSynapse : compõe S sinapses
    MPJRDNeuron --> HomeostasisController : atualiza theta/r_hat
    MPJRDNeuron --> Neuromodulator : calcula R
```

---

## 4. Diagrama de sequência — `forward_pass`

```mermaid
sequenceDiagram
    participant Client as Treinador/Loop
    participant Neuron as MPJRDNeuron
    participant D as MPJRDDendrite[*]
    participant Syn as MPJRDSynapse[*]
    participant H as HomeostasisController
    participant N as Neuromodulator

    Client->>Neuron: forward(x, reward, mode)
    loop para cada dendrito d
      Neuron->>D: forward(x[:, d, :])
      D->>Syn: consulta W (derivado de N)
      Syn-->>D: contribuição sináptica
      D-->>Neuron: v_dend[:, d]
    end

    Neuron->>Neuron: gating/integração (WTA ou cooperativo)
    Neuron->>Neuron: u = soma(v_dend_processado)
    Neuron->>Neuron: spikes = 1[u >= theta]

    alt modo != inference
      Neuron->>H: update(spike_rate)
      H-->>Neuron: theta, r_hat atualizados
    end

    Neuron->>N: forward(reward, spike_rate, saturation_ratio, r_hat)
    N-->>Neuron: R

    opt batch com defer_updates
      Neuron->>Neuron: stats_acc.accumulate(x, gated, spikes)
    end

    Client<<--Neuron: {spikes, u, v_dend, theta, r_hat, R, ...}
```

---

## 5. Decisões arquiteturais

- **Interpretabilidade por estado:** sinapse mantém `N` e `I` explícitos.
- **Separação de escalas temporais:** aquisição online e consolidação/sleep offline.
- **Extensibilidade:** v3.0 reutiliza o core e estende inferência com fase/frequência.
- **Compatibilidade com PyTorch:** módulos `nn.Module`, buffers registrados e operações vetorizadas.
