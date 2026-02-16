# ADR-002 — Integridade por chunk com CRC32C/SHA256 e ECC opcional

## Status
Accepted

## Contexto
Corrupção localizada deve ser detectada sem invalidar o modelo conceitual inteiro.

## Decisão
Aplicar CRC32C + SHA-256 no payload comprimido por chunk e suportar ECC opcional (`none`/`rs(n)`).

## Consequências
- + detecção rápida e robusta de corrupção.
- - overhead de CPU/disco quando ECC está ativo.
