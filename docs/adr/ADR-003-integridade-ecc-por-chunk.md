# ADR-003 — Integridade e ECC por chunk

## Status
Accepted

## Contexto
Arquivos científicos grandes exigem detecção robusta de corrupção e estratégia opcional de correção.

## Decisão
Aplicar por chunk:
- CRC32C (rápido) para detecção imediata;
- SHA-256 (forte) para verificação robusta;
- ECC opcional (`none` ou `rs(n)`) sobre payload comprimido.

Ordem canônica:
- escrita: compressão -> hash/checksum -> ECC -> persistência;
- leitura: leitura -> ECC -> verificação -> descompressão.

## Consequências
### Positivas
- detecção granular e rastreável;
- possibilidade de recuperação em cenários com ECC.

### Trade-offs
- custo adicional de CPU/disco quando ECC está habilitado.

## Dependências
- ADR-002.
