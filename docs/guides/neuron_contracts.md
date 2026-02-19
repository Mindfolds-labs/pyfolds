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
