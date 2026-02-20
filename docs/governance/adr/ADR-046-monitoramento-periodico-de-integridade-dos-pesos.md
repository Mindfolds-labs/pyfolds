# ADR-046 — Monitoramento periódico de integridade dos pesos por SHA-256

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia de Runtime, Segurança de Modelo e Confiabilidade
- **Contexto:** Como evolução da blindagem de checkpoint da `v2.0.2`, foi priorizado um mecanismo em runtime para detectar corrupção silenciosa de pesos durante treinos longos (ex.: bitflips em VRAM).

## Contexto

A estratégia de integridade em checkpoint (`HMAC/SHA-256`) já protege dados em disco e no fluxo de carregamento. Porém, em cenários de execução prolongada, ainda existe janela para corrupção em memória entre checkpoints.

## Decisão

Implementar monitor periódico de integridade em memória com os seguintes pontos:

1. Novo `WeightIntegrityMonitor` em `pyfolds.monitoring.health`.
2. Cálculo de hash `SHA-256` do `state_dict` em intervalos configuráveis (`check_every_n_steps`).
3. Contrato explícito de resultado para cada verificação:
   - `checked=False` fora do intervalo.
   - `checked=True` com `ok=True/False`, hash anterior e hash atual.
4. Exportar o monitor no namespace público de `pyfolds.monitoring`.

## Consequências

### Positivas
- Detecção simples e determinística de alteração inesperada em pesos entre checkpoints.
- Base pronta para integração com `MindControl`, telemetria e alarmes de operação.
- Cobre backlog de hardening para tolerância a falhas em execuções de longa duração.

### Trade-offs
- Overhead proporcional ao tamanho do modelo quando a verificação é executada.
- Hash é calculado em CPU após snapshot dos tensores, portanto o intervalo precisa ser configurado conforme throughput alvo.

## Implementação vinculada

- `src/pyfolds/monitoring/health.py`
- `src/pyfolds/monitoring/__init__.py`
- `tests/unit/core/test_health_monitor.py`

## Referências

- `docs/governance/adr/ADR-045-checkpoints-seguros-com-safetensors-flush-lazy-e-shape-validation.md`
- `docs/governance/quality/issues/ISSUE-009-atualizacao-documentacao-v2-0-2.md`
