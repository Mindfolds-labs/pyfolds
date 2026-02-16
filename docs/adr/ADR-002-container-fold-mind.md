# ADR-002 — Container `.fold/.mind` para checkpoints científicos

## Status
Accepted

## Contexto
`torch.save()` isolado não atende requisitos de inspeção parcial, integridade por seção e metadados auditáveis.

## Decisão
Padronizar container único com extensões semânticas:

- `.fold`: extensão técnica padrão;
- `.mind`: mesmo formato físico com branding IA quando houver chunks `ai_graph` e/ou `ai_vectors`.

Elementos mandatórios:

1. Header fixo com ponteiro para índice.
2. Chunks tipados com compressão opcional.
3. Índice JSON no final do arquivo.
4. Integridade por chunk com CRC32C e SHA-256.
5. ECC opcional por chunk (`none` ou `rs(n)`).

## Consequências
### Positivas
- Leitura parcial eficiente.
- Diagnóstico de corrupção localizado.
- Melhor suporte a auditoria e reprodutibilidade.

### Trade-offs
- Maior complexidade frente ao dump monolítico.
