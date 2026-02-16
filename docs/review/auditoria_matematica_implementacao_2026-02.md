# Auditoria Matemática e de Implementação (2026-02)

## Escopo
Arquivos avaliados nesta rodada:
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/synapse.py`
- `src/pyfolds/core/dendrite.py`
- `src/pyfolds/core/homeostasis.py`
- `src/pyfolds/advanced/stdp.py`
- `src/pyfolds/advanced/backprop.py`

## Síntese executiva
A implementação está globalmente **coerente com um neurônio de plasticidade estrutural com regra three-factor** (pré × pós × neuromodulador), com proteção por domínio (clamps) e limites estruturais (`n_min`, `n_max`).

Foram identificados três pontos de atenção técnico-científica:
1. **Atenuação da força de aprendizado em batch** por normalização de `pre_rate` pelo número de sinapses ativas.
2. **Semântica STDP não canônica** (termos condicionados por spike pós no passo corrente, sem curva explícita Δw(Δt)).
3. **Falhas de testes utilitários de logging** (não matemático, mas com impacto de observabilidade/reprodutibilidade).

---

## 1) Correção matemática das operações

### 1.1 Regra three-factor em `MPJRDSynapse.update`
A dinâmica implementada é:
\[
\Delta I \propto \eta \cdot R \cdot (r_{pre} r_{post}) \cdot (1 + \beta_W W) \cdot dt
\]
com:
- `r_pre`, `r_post` clampados em `[0,1]`
- `R` clampado em `[-1,1]`
- `I` clampado em `[i_min, i_max]`

**Avaliação:** correta no domínio e estável sob parâmetros default. A dinâmica de transição de `N` por limiares de LTP/LTD é finita e limitada por `n_min <= N <= n_max`.

### 1.2 Homeostase
`theta` é atualizado por erro de taxa (`rate - target_spike_rate`) com ganho `homeostasis_eta`, e `r_hat` por EMA:
\[
\hat r_{t+1} = (1-\alpha)\hat r_t + \alpha r_t
\]

**Avaliação:** formulação correta; convergência em média depende de regime de entrada estacionário e `0 < alpha <= 1`.

### 1.3 Mapeamento estrutural para peso
\[
W(N)=\log_2(1+N)/w_{scale}
\]

**Avaliação:** monotônico, saturante e numericamente robusto para `N` inteiro limitado.

---

## 2) Coerência teoria ↔ implementação

### 2.1 Batch learning: normalização de `pre_rate`
No caminho `apply_plasticity`, o código usa:
\[
pre\_rate_i = \frac{x_i \cdot \mathbf{1}(x_i>\tau)}{\max(1, n_{active})}
\]

**Impacto:** o termo Hebbiano efetivo por sinapse cai com `n_active`, mesmo se as taxas absolutas permanecerem elevadas. Isso reduz plasticidade em padrões densos. Não é "errado", mas altera interpretação (taxa absoluta → distribuição relativa).

### 2.2 STDP
No mixin atual, `delta_ltp` e `delta_ltd` são ambos multiplicados por `post_expanded` do passo atual.

**Impacto:** implementação aproxima STDP de um ajuste pós-condicionado por traços, mas não reproduz diretamente a curva canônica pair-based \(\Delta w(\Delta t)\) (Bi & Poo, 1998).

### 2.3 Fluxo de aprendizado e estado
- `ONLINE`: atualizações frequentes, homeostase ativa.
- `BATCH` + `defer_updates`: acumula estatísticas e aplica commit explícito.
- `SLEEP`: consolidação por elegibilidade.
- `INFERENCE`: sem atualização homeostática/plástica.

**Avaliação:** fluxo consistente e bem segmentado por modo.

---

## 3) Estabilidade, domínio e convergência

Condições práticas para operação estável:
- `0 < i_gamma < 1`
- `i_eta * dt` moderado (evitar oscilações entre limiares LTP/LTD)
- `homeostasis_eta <= 0.1` (já sinalizado no código)
- `R` limitado em `[-1, 1]` (já implementado)

Parâmetros críticos:
- `beta_w`: amplifica assimetria entre sinapses fortes/fracas.
- `activity_threshold`: controla densidade ativa e taxa de aprendizado efetiva.
- `ltd_threshold_saturated`: define histerese de sinapses saturadas.
- `consolidation_rate`: acelera/desacelera memória de longo prazo em sono.

---

## 4) Issues propostas (para rastreabilidade)

### Issue A — Revisar normalização de `pre_rate` em BATCH
**Tipo:** ajuste matemático de regra de aprendizado

**Hipótese:** remover divisão por `n_active` (ou tornar configurável) melhora retenção de magnitude hebbiana em entradas densas.

**Aceite sugerido:**
- Benchmark comparando convergência de `spike_rate` e distribuição final de `N` em cenários esparsos vs densos.
- Garantia de domínio (`pre_rate in [0,1]`) preservada.

### Issue B — Teste de curva STDP \(\Delta w(\Delta t)\)
**Tipo:** validação científica

**Hipótese:** comportamento atual diverge do pair-based canônico para certos intervalos temporais.

**Aceite sugerido:**
- Teste que injeta pares pré/pós controlados e verifica sinal de \(\Delta w\) para `Δt>0` e `Δt<0`.
- Documentação explícita se o modelo adotado for "trace-gated" (não canônico).

### Issue C — Corrigir regressões de logger
**Tipo:** confiabilidade/infra de experimento

**Hipótese:** singleton/handlers e níveis de módulo não estão estáveis entre inicializações de teste.

**Aceite sugerido:**
- `tests/unit/utils/test_utils.py` sem falhas no bloco de logging.
- Comportamento determinístico com configuração repetida.

---

## 5) Referências científicas
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type.
- Frémaux, N., & Gerstner, W. (2016). Neuromodulated STDP and three-factor learning rules.
- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. *Neuronal Dynamics*.
- Dayan, P., & Abbott, L. F. *Theoretical Neuroscience*.
