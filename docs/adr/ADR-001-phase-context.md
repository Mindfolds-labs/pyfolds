# ADR-001 — Contexto de fase para coordenação temporal

## Status
Accepted

## Contexto
O modelo precisava de contexto temporal explícito para modular plasticidade, atenção e consolidação.

## Decisão
Adotar fase circadiana (`circadian_phase`) como buffer de runtime e usar bins de fase para observabilidade de atividade.

## Justificativa
Permite controlar mecanismos por fase sem armazenar dados brutos de entrada no módulo.

## Impactos e trade-offs
- Pró: modulação temporal simples e barata.
- Contra: aproximação discreta por bins pode perder dinâmica contínua fina.

## Relação com literatura científica
Compatível com trabalhos de oscillatory coordination e phase coding (theta-gamma como referência conceitual).

## Limitações
Não representa circuitos oscilatórios biológicos completos, apenas gate computacional.

## Próximos passos
Adicionar acoplamento explícito entre bandas e métricas de locking por fase.
