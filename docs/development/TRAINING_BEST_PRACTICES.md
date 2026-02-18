# Boas práticas para códigos de treinamento e neurônio interno (PyFolds)

## 1) Estrutura de arquivos recomendada

- `src/pyfolds/core/`: regras locais do neurônio (sinapse, dendrito, homeostase, neuromodulação).
- `src/pyfolds/layers/`: composição vetorizada de neurônios por camada.
- `src/pyfolds/network/`: orquestração entre camadas e ordenação topológica.
- `src/pyfolds/serialization/`: checkpoint/salvamento com compatibilidade de versão.
- `test/` ou `examples/`: scripts de treino reproduzíveis (ex.: `test/mnist_training.py`).

## 2) Classes envolvidas no fluxo de treino

Fluxo padrão recomendado:
1. `MPJRDConfig`: define hiperparâmetros estruturais e de plasticidade.
2. `MPJRDLayer` + `MPJRDNeuronAdvanced` (opcional): execução neuronal por lote.
3. Modelo `torch.nn.Module` agregador: projeção de entrada + camada MPJRD + cabeça de tarefa.
4. `VersionedCheckpoint`: persistência dos pesos e metadados.
5. `setup_run_logging`: logging em arquivo, sem poluição de terminal.

## 3) Dependências

Dependências mínimas de runtime:
- `torch`, `torchvision`, `numpy`, `zstandard`, `google-crc32c`, `reedsolo`.

Dependências de desenvolvimento/testes:
- `pytest`, `psutil` (testes de stress/performance), `black`, `isort`, `mypy`.

## 4) Padrão de importações

- Preferir **imports absolutos**: `from pyfolds.core import ...`.
- Evitar import dinâmico fora de pontos de fallback explícitos.
- Não acoplar módulos de treino com internals privados de `core`.
- Manter `__init__.py` apenas como fachada de API pública (evitar lógica pesada de runtime).

## 5) Fluxo recomendado de arquitetura

1. Preparar entrada (`[B, N, D, S]`).
2. `forward` em `MPJRDLayer`/`MPJRDNeuron` coletando métricas essenciais.
3. Aplicar loss de tarefa + otimização global (`backward`/`step`).
4. Definir `LearningMode` explícito por fase (`ONLINE`, `INFERENCE`, `SLEEP`).
5. Salvar checkpoints versionados e logs para rastreabilidade.

## 6) Princípios de consistência interna do neurônio

- Garantir device único para buffers internos (evitar CPU/GPU misto).
- Sincronizar métricas de homeostase com ciclo de `commit/sleep`.
- Padronizar thresholds e normalizações de entrada em utilitários compartilhados.
- Evitar escrita de estado fora de métodos de atualização da própria classe.
