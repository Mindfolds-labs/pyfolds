# ADR-007 — Observabilidade e depuração ativável

## Status
Accepted

## Contexto
Era difícil inspecionar conectividade efetiva, poda e atividade por fase.

## Decisão
Criar utilitários:
- `collect_connectivity_snapshot()`
- `collect_pruning_snapshot()`
- `collect_phase_activity_report()`
- `collect_engram_report()`

## Justificativa
Padroniza debug sem persistir artefatos transitórios no `state_dict`.

## Impactos e trade-offs
- Pró: diagnósticos reproduzíveis.
- Contra: snapshots podem custar memória/latência se chamados em excesso.

## Relação com literatura científica
Favorece comparações baseline vs ativo e monitoramento de dinâmicas oscilatórias/replay.

## Limitações
Relatórios ainda agregados; faltam métricas causais e intervenção controlada.

## Próximos passos
Exportadores automáticos para TensorBoard/JSONL por fase.
