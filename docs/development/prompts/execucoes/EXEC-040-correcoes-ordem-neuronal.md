# Prompt Codex — Correções de ordem de execução neuronal (repo principal)

## Contexto
Quero que você aplique correções estruturais no pipeline de `MPJRDNeuronAdvanced` para alinhar ordem de execução com literatura neurocomputacional e eliminar inconsistências entre mixins.

## Objetivos obrigatórios

1. **Refratário como decisão final de spike**
   - Garantir que nenhum mixin posterior reescreva `output['spikes']` de forma a violar bloqueio refratário absoluto.

2. **Adaptação sem conflito de autoridade**
   - Refatorar `AdaptationMixin` para atuar no estado (`u`, corrente adaptativa, limiar efetivo) sem sobrescrever o spike final já validado pelo refratário.

3. **bAP funcional (não apenas observável)**
   - Integrar `dendrite_amplification` no cálculo da dinâmica (ex.: ganho dendrítico, modulação de entrada ou traço), com flag de config para ativar/desativar.

4. **STDP explícito quanto ao sinal pré**
   - Adicionar config `stdp_input_source: "raw" | "stp"`.
   - Se `raw`, STDP deve usar entrada original antes de STP.
   - Se `stp`, manter comportamento atual e documentar.

5. **Semântica temporal unificada**
   - Garantir incremento de tempo em ponto único por passo e compatibilidade entre `BackpropMixin` e `RefractoryMixin`.

6. **LTD canônico configurável**
   - Adicionar `ltd_rule: "classic" | "current"`.
   - `classic`: LTD baseado em `pre_spike * trace_post`.
   - `current`: preservar regra atual para retrocompatibilidade.

7. **Rate checks auditáveis**
   - Evitar clamp silencioso em casos de entrada fora da faixa esperada.
   - Emitir warning estruturado com contexto (modo, dendrito, faixa observada).

## Arquivos-alvo (mínimo)

- `src/pyfolds/advanced/__init__.py`
- `src/pyfolds/advanced/adaptation.py`
- `src/pyfolds/advanced/refractory.py`
- `src/pyfolds/advanced/backprop.py`
- `src/pyfolds/advanced/stdp.py`
- `src/pyfolds/advanced/short_term.py`
- `src/pyfolds/core/neuron.py`
- `src/pyfolds/core/config.py`
- testes em `tests/unit/advanced/` e `tests/unit/core/`
- docs API em `docs/api/advanced/*.md`

## Restrições

- Não quebrar API pública existente sem fallback.
- Manter caminho retrocompatível por flags de config com default atual.
- Sem `try/except` em imports.

## Testes obrigatórios

Rode e garanta verde:

- `pytest tests/unit/advanced/test_refractory.py -q`
- `pytest tests/unit/advanced/test_adaptation.py -q`
- `pytest tests/unit/advanced/test_stdp.py -q`
- `pytest tests/integration/test_neuron_advanced.py -q`

Adicione testes novos para:

- Refratário absoluto inviolável.
- bAP alterando dinâmica funcional (não apenas campo no output).
- `stdp_input_source` (`raw` vs `stp`).
- `ltd_rule` (`classic` vs `current`).
- Consistência temporal entre mixins.

## Entrega

Ao final, entregue:

1. Resumo curto das mudanças.
2. Lista de arquivos alterados.
3. Resultado dos testes executados.
4. Riscos remanescentes.
