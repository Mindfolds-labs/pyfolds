# Plano de Refatoração Arquitetural (sem quebra de API)

## Escopo e fonte
Este plano foi elaborado por inspeção estática do código e dos artefatos de auditoria em `docs/sheer-audit/sheerdocs/code_map.md`.

## 1) Problemas encontrados

### CRÍTICO
1. **Duplicação do conceito de tempo**
   - `time_counter` (mixins avançados) e `global_time_ms` (core) coexistem sem contrato único.
   - Arquivos: `src/pyfolds/advanced/time_mixin.py`, `src/pyfolds/advanced/__init__.py`, `src/pyfolds/core/neuron.py`.

2. **Buffers dependentes de batch no `state_dict`**
   - Buffers como `last_spike_time`, `adaptation_current`, `backprop_trace` mudam shape em runtime e exigem pre-hooks para carga.
   - Isso fragiliza roundtrip entre checkpoints com batch diferente.
   - Arquivos: `src/pyfolds/advanced/refractory.py`, `src/pyfolds/advanced/adaptation.py`, `src/pyfolds/advanced/backprop.py`.

3. **Risco de ordem de mixins impactar semântica científica**
   - Existe validação de MRO, mas o acoplamento entre ordem de herança e semântica de execução permanece alto.
   - Arquivo: `src/pyfolds/advanced/__init__.py`.

### PERFORMANCE
4. **Loops Python em caminho crítico (fallback e atualização)**
   - Há fallback não-vetorizado no `forward` quando `use_vectorized_dendrites=False` e loops por dendrito em updates/plasticidade.
   - Arquivo: `src/pyfolds/core/neuron.py`.

5. **Uso de clones/alocações frequentes em mecanismos avançados**
   - Ex.: cópia de eventos no backprop e estados auxiliares para telemetria/diagnóstico.
   - Arquivo: `src/pyfolds/advanced/backprop.py`.

### ARQUITETURA
6. **Compatibilidade PyTorch/TF parcial e assimétrica**
   - Backend TF limita `integration_mode` para `wta_soft_approx` enquanto Torch suporta múltiplos modos.
   - Contratos existem, mas há divergência funcional por design.
   - Arquivos: `src/pyfolds/tf/neuron.py`, `src/pyfolds/contracts/backends.py`.

## 2) Impacto no sistema
- **Confiabilidade científica:** risco de deriva temporal e resultados não reprodutíveis entre composições de mixins.
- **Confiabilidade operacional:** checkpoints podem falhar/interagir mal ao variar batch shape.
- **Performance:** latência e uso de CPU acima do necessário em cargas grandes.
- **Portabilidade:** paridade Torch↔TF incompleta dificulta validação cruzada e serving híbrido.

## 3) Prioridade de correção
- **P0 (imediato):** tempo unificado, buffers dinâmicos no `state_dict`, garantias de ordem semântica de mixins.
- **P1 (curto prazo):** vetorização e redução de loops/alocações no caminho de execução.
- **P2 (médio prazo):** convergência de contratos Torch/TF e separação formal entre core estável e experimental.

## 4) Arquivos afetados (alvo inicial)
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/advanced/time_mixin.py`
- `src/pyfolds/advanced/__init__.py`
- `src/pyfolds/advanced/refractory.py`
- `src/pyfolds/advanced/adaptation.py`
- `src/pyfolds/advanced/backprop.py`
- `src/pyfolds/tf/neuron.py`
- `src/pyfolds/contracts/backends.py`
- `tests/unit/neuron/test_time_counter.py`
- `tests/unit/neuron/test_contract_conformance.py`
- `tests/unit/test_backend_contracts.py`

## 5) Risco de regressão
- **Alto:** mudanças em tempo e refratário podem alterar comportamento de spike.
- **Médio:** serialização de estado dinâmico pode afetar compatibilidade com checkpoints legados.
- **Médio:** vetorização pode alterar ordem numérica e tolerâncias flutuantes.
- **Baixo/Médio:** organização modular e telemetria têm risco funcional menor, mas risco de observabilidade.

## 6) Estratégia de refatoração

### Fase A — CRÍTICO
1. Definir um **TimeAuthority** único no neurônio base.
2. Manter `time_counter` como alias/delegação para preservar API pública.
3. Migrar buffers batch-dependentes para estado não persistente (ou serialização versionada explícita).
4. Formalizar pipeline de execução dos mecanismos sem depender apenas de MRO.

### Fase B — PERFORMANCE
1. Tornar caminho vetorizado obrigatório por default e restringir fallback com flag de debug.
2. Reduzir clones em estruturas de evento (backprop) com pooling/visões quando possível.
3. Revisar pontos de stack/cat recorrentes em forward/update.

### Fase C — ARQUITETURA
1. Criar camada de contrato unificada Torch/TF com matriz de capacidades declarada.
2. Isolar mecanismos experimentais em pacote com fronteira explícita (`experimental/*`).
3. Expandir telemetria para métricas de custo (latência, memória, overflow de fila) com naming estável.

---

## Níveis de correção (consolidado)

### CRÍTICO
- Bugs científicos
- Bugs de execução
- Buffers quebrados em `state_dict`

### PERFORMANCE
- Vetorização
- Redução de memória
- Eliminação de loops Python em caminho quente

### ARQUITETURA
- Organização de módulos
- Contratos de API
- Melhoria de telemetria

---

## PRs sugeridos

### PR-1 — P0: Centralização temporal sem quebra de API
- **Arquivos:** `core/neuron.py`, `advanced/time_mixin.py`, `advanced/__init__.py`, testes de tempo.
- **Mudanças:** introduzir autoridade única de tempo no core; manter compat (`time_counter`), remover duplicidade lógica.
- **Impacto:** estabilidade científica e previsibilidade entre mixins.
- **Testes necessários:** regressão de `test_time_counter*`, novos testes de equivalência temporal (advanced/wave).

### PR-2 — P0: Hardening de buffers dinâmicos e serialização
- **Arquivos:** `advanced/refractory.py`, `advanced/adaptation.py`, `advanced/backprop.py`, `serialization/*`, testes de roundtrip.
- **Mudanças:** separar estado efêmero por batch do estado persistente; definir política de persistência por buffer.
- **Impacto:** robustez de checkpoint/load em diferentes batch sizes.
- **Testes necessários:** roundtrip com batch variável + backward compatibility de checkpoints antigos.

### PR-3 — P0/P1: Pipeline explícito de mecanismos (substituir acoplamento forte a MRO)
- **Arquivos:** `advanced/__init__.py`, mixins avançados, contrato científico.
- **Mudanças:** executor declarativo de etapas (ordem fixa e auditável), preservando classes públicas atuais.
- **Impacto:** reduz bugs de composição e facilita manutenção.
- **Testes necessários:** snapshot da ordem de execução + testes de não-regressão de spike/refratário/STDP.

### PR-4 — P1: Vetorização total em forward/update
- **Arquivos:** `core/neuron.py`, `core/dendrite*.py`, benchmarks.
- **Mudanças:** remover caminho com loop de dendritos no uso padrão; otimizar atualizações por lote.
- **Impacto:** ganho de throughput e menor overhead em CPU/GPU.
- **Testes necessários:** `tests/performance/*` com metas mínimas e testes de equivalência numérica.

### PR-5 — P2: Matriz de paridade Torch/TF
- **Arquivos:** `tf/neuron.py`, `contracts/backends.py`, docs de contrato.
- **Mudanças:** declarar capacidades suportadas por backend; bloquear explicitamente modos não suportados com erro explicativo e telemetria.
- **Impacto:** previsibilidade de portabilidade e redução de comportamento implícito.
- **Testes necessários:** suíte de conformidade Torch/TF por feature flag + tolerâncias numéricas documentadas.

### PR-6 — P2: Separação Core Estável vs Experimental
- **Arquivos:** `advanced/experimental.py`, estrutura de pacotes, docs/adr.
- **Mudanças:** isolamento de mecanismos experimentais com fronteira clara e toggles versionados.
- **Impacto:** menor risco de regressão no core e governança técnica mais clara.
- **Testes necessários:** smoke tests do core sem experimental + testes A/B dos mecanismos experimentais.
