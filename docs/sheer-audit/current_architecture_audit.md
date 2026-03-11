# Auditoria da Arquitetura Atual do PyFolds

Data: 2026-03-11  
Escopo analisado (inspeção estática): `docs/sheer-audit`, `docs/architecture`, `docs/science`, `docs/mechanisms`, `docs/adr`, `docs/ARCHITECTURE.md`, `docs/STABLE_CORE_ARCHITECTURE.md` e código em `src/pyfolds/{core,advanced,network,contracts,telemetry,tf,wave}`.

---

## 1 Arquitetura real encontrada

### 1.1 Núcleo real (código)
- O núcleo computacional está em `src/pyfolds/core/neuron.py` (`MPJRDNeuron`) com:
  - dendritos (`MPJRDDendrite`),
  - integração dendrítica (`DendriticIntegration`),
  - homeostase (`HomeostasisController`),
  - neuromodulação,
  - acumulação estatística e telemetria.
- Existe variante `MPJRDNeuronV2` em `src/pyfolds/core/neuron_v2.py` com integração cooperativa (soft-like) vetorizada.
- O pipeline avançado é por mixins em `src/pyfolds/advanced/` e composição em `src/pyfolds/advanced/__init__.py`.

### 1.2 Ordem efetiva de mixins avançados
Classe principal: `MPJRDNeuronAdvanced`:
1. `CircadianWaveMixin`
2. `WaveDynamicsMixin`
3. `BackpropMixin`
4. `ShortTermDynamicsMixin`
5. `STDPMixin`
6. `AdaptationMixin`
7. `RefractoryMixin`
8. `MPJRDNeuron`

A ordem é validada em runtime por `_validate_advanced_mro(...)`.

### 1.3 Situação dos mecanismos solicitados
- **RefractoryMixin**: implementado (`advanced/refractory.py`), por amostra `[B]`, com hook para `state_dict`.
- **AdaptationMixin**: implementado (`advanced/adaptation.py`), por amostra `[B]`, com hook para `state_dict`.
- **STDPMixin**: implementado (`advanced/stdp.py`), traços por batch `[B,D,S]`.
- **DendriteIntegration**: implementado como `DendriticIntegration` (`core/dendrite_integration.py`).
- **Homeostasis**: implementado como `HomeostasisController` (`core/homeostasis.py`).
- **WaveOscillator**: implementado (`advanced/wave.py`).
- **ShortTermPlasticity**: implementado como `ShortTermDynamicsMixin` (`advanced/short_term.py`).
- **Inhibition**: implementado em nível de layer (`advanced/inhibition.py`).

### 1.4 Avaliação rápida de custo computacional (hot path)
- Vetorização principal correta em `forward` de neurônio (`einsum`, broadcast e operações tensoriais).
- Custos adicionais por mecanismo:
  - Refractory/Adaptation/STP: baixo a médio (O(BDS) ou O(B)).
  - STDP: médio; vetorizado nos traços, mas com loops Python na etapa de escrita de elegibilidade.
  - Backprop: médio; fila + loops por eventos.
  - Wave/Circadian: médio; cálculos adicionais de fase/envelope e buffers de histórico.
  - Inhibition layer: médio-alto em `n_exc` (matmuls `B x n_exc` e `n_exc x n_exc`).

---

## 2 divergências entre docs e código

1. **Divergência de nomenclatura de mecanismo**: documentos citam “ShortTermPlasticity”, enquanto o código expõe `ShortTermDynamicsMixin`.
2. **Contrato “STDP sem loops Python” não é plenamente verdadeiro**: em `advanced/stdp.py` há loops Python sobre dendritos e fallback por sinapse para aplicar `stdp_eligibility`.
3. **Parte da documentação sugere pipeline formal com contratos de step estritos**, mas o runtime principal ainda opera em dicionários e wrappers de `forward` (sem enforcement universal do contrato formal por passo).
4. **Arquitetura de inibição em docs é tratada como mecanismo disponível**, porém a classe de layer avançada padrão (`MPJRDLayerAdvanced`) não incorpora `InhibitionMixin` automaticamente.

---

## 3 bugs arquiteturais

1. **Bug crítico em `WaveDynamicsMixin.forward`: variável indefinida**
- Em `src/pyfolds/advanced/wave.py`, bloco de métricas experimentais usa `phase_sync_mean`, mas a variável não é definida no método.
- Resultado esperado: `NameError` quando `enable_experimental_coherence_metrics=True`.

2. **Quebra de buffer registrado em `WaveMixin._update_sync_memory`**
- Em `src/pyfolds/advanced/wave.py`, `self.phase_ptr` (buffer registrado) é sobrescrito por reatribuição (`self.phase_ptr = torch.tensor(...)`) em vez de atualização in-place.
- Isso pode remover o buffer do registro esperado do módulo, impactando `state_dict` e consistência de device.

3. **Atualização de ponteiro sem device explícito em `WaveDynamicsMixin.forward`**
- Em `self.phase_pointer.copy_(torch.tensor((ptr + 1) % self.cfg.phase_buffer_size))` o tensor criado é CPU por default.
- Em execução GPU, há risco de erro de device mismatch.

4. **Semântica temporal conflitante no contador global**
- `TimedMixin` incrementa `time_counter` por passo (`_increment_time_once`).
- `CircadianWaveMixin._advance_circadian` também força `time_counter = age_seconds`.
- Isso não duplica o buffer, mas mistura duas semânticas de tempo (tempo de simulação vs idade circadiana), criando risco de inconsistência em STDP/backprop/refractory dependentes de `time_counter`.

---

## 4 problemas de escalabilidade

1. **STDP parcialmente não vetorizado no commit de elegibilidade**
- Cálculo de traços é vetorizado, mas persistência em dendritos/sinapses ainda usa loops Python (principalmente fallback legado por sinapse), limitando escala para topologias grandes.

2. **Inibição lateral com matriz densa `n_exc x n_exc`**
- Em `InhibitionLayer`, o kernel lateral e multiplicação densa podem crescer quadraticamente para camadas muito grandes.

3. **Pipeline wave/circadiano adiciona múltiplos payloads e históricos**
- Para execuções longas com telemetria ampla, há risco de overhead acumulado se não houver amostragem/limites agressivos.

---

## 5 problemas de consistência científica

1. **LTD no STDP depende de regra configurável não-clássica por padrão**
- `ltd_rule="current"` aplica LTD gated por spike pós (legado), o que diverge da formulação clássica pre/post estrita.
- É uma decisão válida de engenharia, mas precisa estar explicitamente diferenciada de STDP biológico clássico em toda documentação de ciência.

2. **Mecanismos wave/speech são experimentais e parcialmente heurísticos**
- ADRs reconhecem simplificações (gammatone aproximado, PAC por lote, kernel espacial isotrópico), então equivalência biológica forte não é suportada.

3. **Uso de fase média e resets de fase por evento**
- Cientificamente útil como proxy de entrainment, mas pode induzir interpretação excessiva se não for tratado como mecanismo fenomenológico.

---

## 6 problemas de implementação

### 6.1 Revisão por mecanismo (variáveis, custo, dependências, integração, conflitos)

#### RefractoryMixin
- Variáveis: `last_spike_time`, `t_refrac_abs`, `t_refrac_rel`, `refrac_rel_strength`.
- Custo: baixo (O(B)).
- Dependências: `time_counter`, `homeostasis`, `AdaptationMixin` opcional.
- Integração: aplica pós-`super().forward` e reescreve `spikes` finais.
- Conflitos: depende de `theta`/`theta_eff` com shape compatível; valida shape, bom.

#### AdaptationMixin
- Variáveis: `adaptation_current`, `adaptation_increment`, `adaptation_tau`, etc.
- Custo: baixo (O(B)).
- Dependências: saída somática (`u_raw`) e spikes finais.
- Integração: antes do threshold efetivo no refratário.
- Conflitos: sem conflito crítico observado; possui pre-hook para `state_dict`.

#### STDPMixin
- Variáveis: `trace_pre`, `trace_post`, parâmetros `A_plus/A_minus/tau_*`.
- Custo: médio (O(BDS) + loops de escrita).
- Dependências: `cfg`, `dendrites`, `synapse_batch` opcional.
- Integração: executa após `super().forward` usando spikes resultantes.
- Conflitos: não quebra ordem, mas contradiz requisito “sem loops Python” no commit final de elegibilidade.

#### DendriticIntegration
- Variáveis: sem estado mutável pesado; usa `cfg`.
- Custo: baixo-médio (sigmoid + normalização por batch).
- Dependências: `v_dend`, `theta`.
- Integração: componente limpo e modular.
- Conflitos: não observados.

#### HomeostasisController
- Variáveis: buffers `theta`, `r_hat`, `integral_error`, histórico de estabilidade.
- Custo: baixo.
- Dependências: taxa de spike.
- Integração: chamada no core e no refratário (defer para pós-bloqueio).
- Conflitos: coexistência está correta para cadeia avançada.

#### WaveOscillator / WaveDynamicsMixin
- Variáveis: `phase`, caches sin/cos, históricos de fase, `wave_time`.
- Custo: médio.
- Dependências: modo de aprendizado, `time_counter`, opcional áudio e coordenadas.
- Integração: pós-forward da classe base.
- Conflitos: bugs de implementação (variável indefinida, ponteiro/buffer/device) e duplicidade de métodos (`_compute_phase_coherence` definido duas vezes em `WaveDynamicsMixin`).

#### ShortTermDynamicsMixin (ShortTermPlasticity)
- Variáveis: `u_stp`, `R_stp`.
- Custo: médio (O(BDS) para pre_spike mean + O(DS) updates).
- Dependências: input `x` e config.
- Integração: modula entrada antes de `super().forward`.
- Conflitos: sem conflito crítico encontrado.

#### Inhibition
- Variáveis: `W_E2I`, `W_I2E`, `lateral_kernel`, `inh_potential`, `inh_threshold`.
- Custo: médio-alto (matrizes densas).
- Dependências: saída excitatória (`spikes`, `u`, `theta`).
- Integração: layer-level, não neuron-level.
- Conflitos: se `theta`/`u` vierem em formas não esperadas, há validações e exceptions (bom), mas integração automática com layer advanced não é padrão.

### 6.2 Buffers PyTorch
- Há uso consistente de `register_buffer` para estados críticos.
- Pontos de risco:
  - reatribuição de buffer (`phase_ptr`) em vez de `copy_/fill_`;
  - criação de tensores sem device explícito em caminhos que copiam para buffers possivelmente em GPU.

### 6.3 state_dict
- Adaptation/Refractory/Backprop possuem pre-hooks para resize e melhor compatibilidade.
- Em contrapartida, `trace_pre`/`trace_post` do STDP não são buffers/params; portanto não entram em `state_dict` (ok se intencional, mas deve ser explicitado no contrato de checkpoint científico).

### 6.4 Vetorização
- Núcleo e grande parte dos mecanismos estão vetorizados.
- STDP ainda tem trecho não totalmente vetorizado na persistência em elegibilidade.

### 6.5 STDP sem loops Python
- **Não atendido integralmente** devido loops em `STDPMixin._update_stdp_traces` no commit para dendritos/sinapses.

### 6.6 time_counter duplicado
- **Buffer não está duplicado**, graças ao `TimedMixin._ensure_time_counter`.
- **Semântica pode conflitar** por sobrescrita em circadiano (`time_counter <- age_seconds`) além do incremento por passo.

---

## 7 riscos de manutenção

1. **Alto risco de regressão no módulo wave**
- Arquivo concentra classes e responsabilidades distintas (oscilador, mixin wave, mixin wave dynamics, integração de speech tracking).

2. **Acoplamento implícito entre mixins por chaves de dicionário**
- A cadeia depende de campos como `u`, `u_raw`, `theta`, `theta_eff`, `spikes`; mudanças locais podem quebrar mixins downstream.

3. **Contratos formais vs runtime real ainda não unificados**
- A lacuna entre especificação e validação em tempo de execução dificulta governança arquitetural de longo prazo.

4. **Semântica temporal multifonte**
- `time_counter`, `wave_time`, `age_seconds` coexistem com papéis parcialmente sobrepostos.

---

## 8 recomendações

### Prioridade P0 (corrigir imediatamente)
1. Corrigir `phase_sync_mean` indefinido em `WaveDynamicsMixin.forward`.
2. Corrigir atualização de `phase_ptr` e `phase_pointer` para operações in-place e device-safe.
3. Remover duplicidade de método `_compute_phase_coherence` em `WaveDynamicsMixin`.

### Prioridade P1 (arquitetura)
4. Definir contrato explícito de tempo:
   - `time_counter` = tempo de simulação,
   - `age_seconds` = relógio circadiano,
   - sem sobrescrita cruzada silenciosa.
5. Formalizar no docs de checkpoint quais estados são persistidos (`state_dict`) e quais são transitórios (ex.: `trace_pre/post`).
6. Tornar STDP fully vectorized também no commit de elegibilidade (eliminar loop por sinapse no caminho principal).

### Prioridade P2 (governança e escalabilidade)
7. Reduzir acoplamento por dicionário na cadeia de mixins (tipagem/contrato de saída por etapa).
8. Para inibição em escala grande, oferecer opção esparsa/bloqueada para `lateral_kernel`.
9. Atualizar documentação para refletir nomes reais de classes e limitações científicas/heurísticas dos mecanismos experimentais.

---

## Conclusão executiva
A arquitetura real está majoritariamente alinhada com o design modular descrito (núcleo + mixins + telemetria), com boa cobertura de mecanismos e uso consistente de buffers PyTorch. Contudo, há divergências relevantes entre documentação e implementação em pontos críticos: STDP ainda não está totalmente livre de loops Python, o módulo wave possui bugs concretos de execução/estado, e a semântica de tempo precisa unificação para evitar efeitos colaterais entre mecanismos. Esses pontos devem ser corrigidos antes de qualquer reescrita arquitetural ampla.
