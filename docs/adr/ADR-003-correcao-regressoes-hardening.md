# ADR-003 — Correção de regressões e hardening inicial

## Status
Accepted

## Contexto
Foram observadas falhas de regressão em refratário, serialização e estabilidade de API, reduzindo confiança da suíte e previsibilidade do runtime.

## Decisão
Consolidar hardening no Sprint 0:

1. Ajustar semântica do refratário relativo.
2. Corrigir fallback de checksum para CRC32C Castagnoli correto.
3. Limpar ambiguidade de API legada em camadas.
4. Tornar testes ECC resilientes a dependências opcionais.
5. Fortalecer fechamento de recursos e escrita robusta em serialização.

## Consequências
### Positivas
- Comportamento alinhado com documentação e testes.
- Menor sensibilidade a variações de ambiente.
- Base mais segura para evolução do formato.

### Trade-offs
- Possível impacto em integrações dependentes de APIs legadas.
