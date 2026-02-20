# ADR-044 — MindControl: cérebro externo e mutação runtime segura

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia de Runtime, Telemetria e Core
- **Contexto:** A ISSUE-013 exige um ciclo fechado desacoplado para reagir à telemetria e aplicar mutações de parâmetros sem interromper o loop de treino e sem corromper o grafo de autograd do PyTorch.

## Contexto

O core já possui telemetria assíncrona com baixo overhead e emissão em fase de `commit`. Faltava um mecanismo explícito para:

1. consumir eventos de telemetria como sinal de controle;
2. decidir mutações de runtime com limites de segurança;
3. injetar alterações somente em fronteiras seguras de execução (`forward`/`apply_plasticity`).

Sem esse desenho, mutações agressivas poderiam gerar NaN, instabilidade de taxa de disparo e, no pior caso, erro de operação in-place em tensores que participam do autograd.

## Decisão

Adotar arquitetura MindControl em malha fechada desacoplada com dois vetores simplex:

1. **Leitura:** `MindControlSink` escuta eventos `commit` da telemetria.
2. **Decisão:** `MindControlEngine` analisa `spike_rate`/`post_rate` e produz comandos de mutação.
3. **Injeção:** comandos são enfileirados em `MutationQueue` e aplicados pelo neurônio no início de fronteiras seguras.

Além disso, manter compatibilidade com o controlador já existente (`MindControl` + `decision_fn`) para integrações prévias.

## Regras de segurança

- **Zero-crash policy:** falhas no plano de controle não interrompem o plano de dados (treino).
- **Graph-safety:** atualização de tensores continua no core com `torch.no_grad()` e `copy_()`.
- **Boundary checking:** `MindControlEngine` aplica `safety_bounds` para parâmetros críticos.

## Consequências

### Positivas
- Controle adaptativo em tempo real sem lock no caminho crítico de cálculo.
- Menor acoplamento entre telemetria e lógica de decisão.
- Trilho formal para novas políticas (dormência, hiperatividade, drift de homeostase).

### Trade-offs
- Mais estados em runtime (fila por neurônio e mapa de registro no engine).
- Necessidade de testes de integração específicos para mutação e autograd.

## Implementação vinculada

- `src/pyfolds/monitoring/mindcontrol.py`
- `src/pyfolds/core/neuron.py`
- `tests/integration/test_mindcontrol_runtime.py`

## Referências

- `docs/development/prompts/relatorios/ISSUE-013-mindcontrol.md`
- `docs/development/prompts/execucoes/EXEC-013-mindcontrol.md`
