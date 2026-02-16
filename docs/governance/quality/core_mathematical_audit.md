# Auditoria técnico-científica do core MPJRD

## Escopo
Revisão dos módulos centrais (`neuron`, `dendrite`, `synapse`, `homeostasis`, `neuromodulation`, `accumulator`) com foco em:
- coerência entre equações e implementação;
- estabilidade/convergência de dinâmicas;
- validade de domínio e parâmetros críticos.

## Principais achados

### 1) Especificidade sináptica na regra Hebbiana (corrigido)
**Achado:** `MPJRDDendrite.update_synapses_rate_based` encaminhava o mesmo vetor `pre_rate` inteiro para cada sinapse, levando a atualização quase idêntica entre sinapses (colapso da localidade da regra `pre_i * post`).

**Impacto matemático:** viola o princípio de localidade da plasticidade hebbiana e distorce seletividade sináptica.

**Ação aplicada:** cada sinapse agora recebe seu `pre_rate[i]` local.

### 2) Estados de curto prazo (u, R) no core de dendrito (corrigido)
**Achado:** o cache do dendrito assumia `syn.u` e `syn.R`, estados que não pertencem à classe core `MPJRDSynapse`.

**Impacto funcional:** risco de falha em tempo de execução no forward.

**Ação aplicada:** estados STP ficaram opcionais no cache (somente quando backend os disponibiliza) e mensagens explícitas orientam uso de `ShortTermDynamicsMixin`.

### 3) Homeostase: estabilidade local
A dinâmica de limiar é um controle proporcional simples:
\[
\theta_{t+1} = \theta_t + \eta_h (r_t - r^*) + \text{rescue}(r_t)
\]
com `clamp` em `[theta_min, theta_max]`.

**Leitura técnica:** estável para `homeostasis_eta` pequeno (como já indicado por warning em config), mas pode oscilar com ganhos altos e estimativa ruidosa de taxa.

### 4) Neuromodulação `surprise`
\[
R = \text{clip}(b + k\,|r_t - \hat r_t|, -1, 1)
\]

**Observação:** por construção o termo de surpresa é não-negativo; LTD modulada por `R<0` depende do viés `b<0` ou de modo externo. Isso é coerente com modelos onde surpresa só aumenta plasticidade, mas restringe regime bidirecional.

## Parâmetros críticos
- `i_eta`, `i_gamma`, `beta_w`: ganho efetivo de atualização de `I`.
- `homeostasis_eta`, `homeostasis_alpha`: compromisso entre velocidade de adaptação e oscilações.
- `cap_k_sat`, `cap_k_rate`, `sup_k`: sensibilidade do neuromodulador.
- `ltd_threshold_saturated`, `saturation_recovery_time`: histerese de saturação.

## Referências científicas de suporte
- Markram et al., 1998 (STP/facilitação-depressão).
- Tsodyks & Markram, 1997 (dinâmica de recursos sinápticos).
- Gerstner & Kistler, *Spiking Neuron Models* (homeostase e regras locais).
- Sutton & Barto, *Reinforcement Learning* (papel de sinais moduladores globais).
