# ADR-005 — Replay e consolidação de sono

## Status
Accepted

## Contexto
Replay offline deve reforçar memória e opcionalmente consolidar poda.

## Decisão
Permitir consolidação de poda após replay com flag `consolidate_pruning_after_replay`.

## Justificativa
Separa reativação de memória da decisão estrutural final de pruning.

## Impactos e trade-offs
- Pró: fluxo explícito e controlável.
- Contra: depende de thresholds heurísticos.

## Relação com literatura científica
Alinhado com literatura de replay/consolidation como mecanismo offline de refinamento.

## Limitações
Não modela estágios de sono (NREM/REM) separadamente.

## Próximos passos
Estratificar replay por fase e tipo de engrama.
