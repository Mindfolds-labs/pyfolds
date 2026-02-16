# ADR-002 — Container `.fold/.mind` para checkpoints científicos

## Status
Accepted

## Contexto
`torch.save()` isolado não atende bem inspeção parcial, metadados auditáveis e integridade granular.

## Decisão
Padronizar um container único com duas extensões:
- `.fold`: formato técnico;
- `.mind`: mesma estrutura física, com semântica de produto/IA.

Layout:
1. header fixo com `magic`, `index_off`, `index_len`;
2. sequência de chunks tipados;
3. índice JSON final com offsets e hashes.

## Consequências
### Positivas
- leitura parcial por chunk;
- metadados unificados de reprodutibilidade;
- diagnóstico melhor de corrupção localizada.

### Trade-offs
- maior complexidade comparado a arquivo único `torch.save`.

## Dependências
- ADR-001.
