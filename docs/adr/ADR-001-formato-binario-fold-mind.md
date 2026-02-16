# ADR-001 — Formato binário `.fold/.mind` com índice no trailer

## Status
Accepted

## Contexto
Era necessário um formato que suportasse retomada de treino, inspeção parcial e evolução incremental sem depender de desserialização integral.

## Decisão
Adotar um container binário com:
- header fixo (`magic`, `header_len`, `index_off`, `index_len`);
- chunks tipados com metadados de tamanho e integridade;
- índice JSON no final do arquivo com offsets e hashes por chunk.

## Consequências
### Positivas
- leitura parcial por offset;
- maior observabilidade e auditabilidade;
- compatibilidade progressiva via inclusão de novos chunks.

### Trade-offs
- maior complexidade que `torch.save` isolado;
- necessidade de política explícita de validação em leitores.

## Relacionamentos
- Complementado por [ADR-002](./ADR-002-compressao-zstd-por-chunk.md), [ADR-003](./ADR-003-ecc-opcional-por-chunk.md) e [ADR-004](./ADR-004-validacao-multicamada.md).
