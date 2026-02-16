# ADR-007 — Monitoramento periódico de invariantes (`N`, `I`, `theta`)

## Status
Proposed

## Contexto
Sem monitoramento contínuo, degradações silenciosas de estado interno podem só aparecer tardiamente em métricas finais.

## Decisão
Introduzir monitor de saúde com checagens periódicas de invariantes:
- domínio válido e limites esperados para `N`, `I`, `theta`;
- métricas de violação por janela temporal;
- integração com telemetria para diagnóstico.

## Consequências
- detecta drift e inconsistências cedo;
- adiciona pequeno custo de execução e armazenamento de sinais de saúde.

## Dependências
- Complementa [ADR-006](./ADR-006-safe-weight-law.md).
- Fornece insumo para testes de [ADR-009](./ADR-009-testes-de-propriedades-matematicas.md).
