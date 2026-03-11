# Contratos de execução do neurônio (Torch/TF)

Este guia define um **contrato canônico** para um passo temporal do neurônio e
como as implementações de backend (Torch e TensorFlow) devem se alinhar.

## Ordem obrigatória dos mecanismos

A ordem invariável do passo é:

1. STP
2. integração
3. SFA
4. threshold
5. refratário
6. bAP
7. STDP
8. homeostase

Essa sequência está formalizada em `pyfolds.contracts.CONTRACT_MECHANISM_ORDER`.


## Ordem canônica de contrato vs ordem efetiva de wrappers (neurônio avançado)

Para evitar ambiguidades, distinguimos dois níveis:

1. **Ordem canônica de contrato (`contracts/`)**
   - Sequência normativa e estável dos mecanismos no passo temporal (STP → integração → SFA → threshold → refratário → bAP → STDP → homeostase).
   - Referência principal para conformidade de backend e testes de contrato.

2. **Ordem efetiva de wrappers no neurônio avançado (`advanced/__init__.py` + mixins)**
   - A MRO define wrappers que executam parte da lógica no pré-`super()` e parte no pós-`super()`.
   - Em termos práticos:
     - `ShortTermDynamicsMixin` atua no pré-`super()` (modula entrada).
     - `RefractoryMixin` atua no pós-`super()` para decisão final de spike pós-máscara refratária.
     - `STDPMixin` atualiza plasticidade no pós-`super()`, consumindo spike final do output.
     - `AdaptationMixin` delega o cálculo de decisão/atualização de spike para o ponto de decisão no refratário (via hooks/métodos auxiliares).

Regra de leitura: a ordem canônica define **semântica de contrato**; a ordem de wrappers define **pontos de aplicação no código** sem alterar a autoridade do spike final pós-refratário.

## Forma de entrada e saída

- Entrada: `NeuronStepInput(x, dt, time_step)`
  - `x`: tensor no formato `[B, D, S]`
  - `dt`: delta temporal do passo
  - `time_step`: tempo corrente no início do passo
- Saída: `NeuronStepOutput(spikes, somatic, step_trace)`
  - `spikes`: tensor `[B]`
  - `somatic`: tensor `[B]`
  - `step_trace`: trilha de ordem + semântica temporal

## Semântica de `time_step`

`time_step` deve permanecer constante durante a execução dos mecanismos e ser
incrementado **somente ao final do passo** (`time_step_after = time_step_before + dt`).

## Divergências numéricas aceitáveis entre backends

Para testes de conformidade Torch vs TF com `float32`, usar:

- `atol = 1e-6`
- `rtol = 1e-5`

Essas tolerâncias cobrem diferenças de arredondamento esperadas de kernels e
ordem de redução numérica.


## Contratos finais de release (alinhamento com ADR)

- Trilha `INFERENCE`: saída deve permanecer compatível e auditável (`spikes`, `somatic`, e interfaces consumidoras com `u_values`/`u` quando aplicável em camada).
- Trilha `ONLINE`: preservar ordem de mecanismos do contrato e consistência temporal (`time_step` incrementado ao fim).
- Trilha `BATCH`: preservar invariância de escala de atualização por lote (sem dependência espúria do tamanho do batch).
- Trilha `SLEEP`: replay e consolidação de pruning são desacoplados; consolidação só ocorre com flag explícita.
