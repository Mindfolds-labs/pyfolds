# README — Auditoria de Ordem de Execução Neuronal (MPJRD Advanced)

Este documento consolida os achados da auditoria sobre **ordem de execução** dos mecanismos neurais avançados e propõe um plano de correção alinhado à literatura.

## Escopo auditado

- `src/pyfolds/advanced/__init__.py`
- `src/pyfolds/advanced/backprop.py`
- `src/pyfolds/advanced/short_term.py`
- `src/pyfolds/advanced/stdp.py`
- `src/pyfolds/advanced/adaptation.py`
- `src/pyfolds/advanced/refractory.py`
- `src/pyfolds/core/neuron.py`

---

## 1) Ordem atual observada no código

A classe `MPJRDNeuronAdvanced` declara os mixins nesta ordem:

1. `BackpropMixin`
2. `ShortTermDynamicsMixin`
3. `STDPMixin`
4. `AdaptationMixin`
5. `RefractoryMixin`
6. `MPJRDNeuronBase`

No fluxo real com `super()`, o `forward()` passa pelos mixins nessa cadeia e depois retorna “desempilhando” as pós-operações.

---

## 2) Achados técnicos confirmados

### Achado A — adaptação recalcula spikes após estágio refratário

`AdaptationMixin.forward()` aplica corrente de adaptação e **recalcula `spikes`** com `u_adapted >= theta`. Isso ocorre após `super().forward(...)` e altera `output['spikes']` no retorno do mixin. Se a sequência entre mixins não for cuidadosamente controlada, o sinal final pode violar a intenção de bloqueio refratário.

### Achado B — bAP calculado, mas sem acoplamento explícito no somatório dendrítico

`BackpropMixin` calcula `dendrite_amplification`, agenda eventos e devolve esse tensor no dicionário de saída. Porém, o `forward()` base (`MPJRDNeuron.forward`) integra dendritos diretamente a partir de `x` sem consumir esse ganho no cálculo de `v_dend/u`.

### Achado C — STDP usa `x` já modulado por STP

`ShortTermDynamicsMixin.forward()` modula `x` e chama `super()` com `x_modulated`; em seguida `STDPMixin.forward()` usa o `x` recebido para detectar `pre_spikes`. Na prática, STDP está observando sinal pré já transformado por STP, o que pode ser desejável em alguns modelos, mas deve ser decisão explícita.

### Achado D — atualização temporal no início do backprop

`BackpropMixin.forward()` incrementa `time_counter` no início e processa fila de backprop antes do `forward` base. O `RefractoryMixin` também usa o mesmo `time_counter` para cálculo do período refratário.

### Achado E — regra LTD implementada difere de formulação STDP clássica

No STDP atual, `delta_ltd` usa `trace_post * post_spike` (broadcast). Em formulações clássicas de pareamento, LTD costuma depender do evento pré com memória pós (`pre_spike` × `trace_post`).

### Achado F — checks de taxa no núcleo podem mascarar problemas de entrada

No `core/neuron.py`, existe clamp em `pre_rate` e um guard de `spike_rate` muito permissivo. Isso evita crash, mas pode esconder entrada não normalizada e dificultar diagnóstico.

---

## 3) Ordem recomendada (literatura + engenharia)

Ordem sugerida por passo de simulação:

1. **Atualização temporal global** (`t <- t + dt`) com política única.
2. **Pré-sináptico**: STP sobre entrada (facilitação/depressão).
3. **Integração dendrítica/somática**: cálculo de `v_dend`, `u` e spike bruto.
4. **Refratário absoluto/relativo**: bloqueio e/ou boost de limiar.
5. **Adaptação (SFA)** aplicada de forma consistente ao estado que alimenta o próximo passo (evitar recomputar spike em conflito com refratário).
6. **STDP** com definição explícita do par de eventos (`pre/post`) e escolha documentada entre usar sinal bruto ou modulado por STP.
7. **bAP**: aplicar ganho onde ele efetivamente altera o caminho dendrítico/plástico, não apenas telemetria.

Referências-base usuais: Hodgkin & Huxley (1952), Bi & Poo (1998), Benda & Herz (2007), Markram et al. (1998), Magee & Johnston (1997).

---

## 4) Plano objetivo de correção

1. **Tornar o refratário “autoridade final” do spike**.
2. **Mudar adaptação para atuar no estado interno (u/threshold) sem sobrescrever spike final bloqueado**.
3. **Acoplar `dendrite_amplification` no caminho de cálculo (entrada ou ganho dendrítico)**.
4. **Escolher e documentar STDP sobre `x_raw` ou `x_stp` (com flag de config)**.
5. **Unificar semântica temporal (um único ponto de incremento por passo)**.
6. **Revisar LTD para forma canônica configurável (`ltd_mode='classic'|'current'`)**.
7. **Substituir clamps silenciosos por validação + logging estruturado em modo debug/auditoria**.

---

## 5) Critérios de aceite (antes de merge)

- Teste de regressão: spike não pode escapar de refratário absoluto.
- Teste de integração: bAP deve alterar métrica funcional (`u`, `v_dend` ou taxa), não apenas campo de saída.
- Teste A/B: STDP com `x_raw` vs `x_stp` deve produzir diferenças reprodutíveis e documentadas.
- Teste temporal: `time_counter` com progressão sem offset inesperado entre refratário e backprop.
- Teste de robustez: warning explícito ao detectar entrada fora da faixa esperada sem mascarar silenciosamente.

---

## 6) Entregáveis esperados

- PR técnico com refactor mínimo dos mixins avançados.
- Atualização de docs em `docs/api/advanced/*.md` explicando a nova ordem.
- Nova suíte de testes em `tests/unit/advanced/` cobrindo os cenários acima.
