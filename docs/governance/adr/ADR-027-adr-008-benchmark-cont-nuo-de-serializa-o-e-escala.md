# ADR-008 — Benchmark contínuo de serialização e escala

## Status
Accepted

## Contexto
Trade-offs entre compressão, ECC e throughput impactam uso em modelos grandes e pipelines de CI/CD.

## Decisão
Instituir benchmark recorrente para:

- serialização/desserialização por faixa de tamanho;
- impacto de ECC e compressão na latência;
- consumo de memória em leitura de artefatos grandes.

Resultados devem virar baseline versionada para comparação entre releases.

## Consequências
### Positivas
- Decisões de performance baseadas em dados.
- Detecção precoce de regressões.

### Trade-offs
- Necessidade de infraestrutura e rotina de medição contínua.
