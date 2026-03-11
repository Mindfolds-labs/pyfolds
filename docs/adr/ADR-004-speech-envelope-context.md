# ADR-004 — Contexto de envelope de fala/áudio

## Status
Accepted

## Contexto
O projeto precisa acomodar sinais temporais (ex.: envelope de fala) sem persistir streams brutos.

## Decisão
Manter datasets/áudio/tokens longos fora do módulo; no módulo ficam apenas estados compactos de runtime.

## Justificativa
Reduz memória persistente e evita poluição de `state_dict`.

## Impactos e trade-offs
- Pró: serialização limpa.
- Contra: exige pipeline externo bem definido.

## Relação com literatura científica
Coerente com cortical tracking abstraído por features compactas e não pelo waveform bruto completo.

## Limitações
Não há implementação explícita de extração de envelope neste patch.

## Próximos passos
Adicionar encoder externo dedicado de envelope com interface clara.
