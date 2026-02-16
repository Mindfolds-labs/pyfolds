# ADR-001 — Container único `.fold/.mind`

## Status
Accepted

## Contexto
Precisamos de um artefato de checkpoint inspecionável, com leitura parcial e metadados auditáveis.

## Decisão
Adotar container binário único para `.fold` e `.mind`, com header fixo, chunks tipados e índice JSON final.

## Consequências
- + melhor inspeção operacional e científica.
- - maior complexidade que `torch.save` simples.
