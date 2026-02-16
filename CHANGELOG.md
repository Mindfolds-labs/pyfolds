# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

## [3.0.0-dev] - 2026-02-15

### Added
- Testes unitários adicionais para `StatisticsAccumulator` cobrindo validação de shape, acúmulo de métricas extras e reset com histórico.
- Testes unitários adicionais para `MPJRDSynapse` cobrindo filtro por atividade, `plastic=False`, multiplicador por modo, recuperação de saturação e consolidação.
- Arquitetura MPJRD-Wave com codificação por fase/frequência.
- Documentação de fase, latência e integração cooperativa.
- Diagramas técnicos adicionais (classes core/wave, estado da sinapse).

### Changed
- Documento de estratégia de testes com foco incremental para PRs de 1 dia.
- Organização da documentação em categorias (`blueprint`, `api`, `guides`, `research`, `diagrams`).
- README transformado em portal de entrada para documentação.

## [2.0.0] - 2025-XX-XX

### Added
- Núcleo MPJRD com sinapses de filamentos discretos (`N`) e potencial interno (`I`).
- Homeostase adaptativa e neuromodulação configurável.
- Pipeline de consolidação offline (sleep).
