<div align="center">

# 🧠 PyFolds

[![PyPI](https://img.shields.io/badge/PyPI-pyfolds-blue)](https://pypi.org/project/pyfolds/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-Organized-success)](docs/README.md)

Framework neurocomputacional bioinspirado para computação dendrítica não-linear, plasticidade estrutural e codificação por fase/frequência.

</div>

---

## Visão geral

O PyFolds implementa o modelo MPJRD (v2.0) e sua extensão MPJRD-Wave (v3.0), com pipeline explícito:

**Sinapse (`N`, `I`, `W`) → Dendrito (`v_dend`) → Soma (`u`, `theta`) → Saída (`spikes` ou `wave`)**.

Isso facilita auditoria e pesquisa porque os estados internos são interpretáveis e mensuráveis.

## Instalação

```bash
pip install pyfolds
```

Para desenvolvimento:

```bash
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```


## Benchmarks de serialização

Para medir throughput de escrita/leitura e taxa de compressão do formato `.fold`:

```bash
python scripts/run_benchmarks.py --output docs/assets/benchmarks_results.json
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/BENCHMARKS.md
```

Interpretação rápida:
- **throughput (MiB/s)**: quanto maior, melhor.
- **razão de compressão vs `none`**: valores menores que `1.0` indicam arquivo comprimido menor. O método pode ser `fold:zstd` (quando disponível) ou fallback `zlib(level=6)`.
- O workflow `.github/workflows/benchmarks.yml` executa semanalmente e atualiza os artefatos automaticamente.

## 📚 Portal de Documentação

Acesso rápido aos ativos de conhecimento e especificações do projeto:

- 📑 **[Índice Geral](docs/README.md)**: Mapa completo de navegação.
- 🧪 **[Lógica Científica](docs/SCIENTIFIC_LOGIC.md)**: Fundamentação teórica do modelo MPJRD.
- 🏗️ **[Arquitetura](docs/ARCHITECTURE.md)**: Diagramas C4, sequência e Wave v3.0.
- 📦 **[Protocolo .fold/.mind](docs/FOLD_SPECIFICATION.md)**: Especificação de serialização binária.
- 🔌 **[Referência de API](docs/API_REFERENCE.md)**: Documentação funcional de módulos e classes.
- 🧾 **[Registro de ADRs](docs/adr/INDEX.md)**: Histórico de decisões arquiteturais.
- 📈 **[Relatório de Benchmarks](docs/BENCHMARKS.md)**: Métricas de performance e compressão.

---

## 🛠️ Manuais de Implementação

Nossa documentação é segmentada por perfil de atuação para otimizar o tempo de busca (UX):

### 🚀 Para Desenvolvedores
* **Quickstart**: [Instalação e conceitos básicos](docs/guides/QUICKSTART.md).
* **Guia MNIST**: [Exemplo prático de visão computacional](docs/guides/mnist_example.md).
* **Tutorial Wave**: [Implementação de dinâmica de fase](docs/guides/wave_tutorial.md).
* **Exemplos**: [Repositório de códigos de referência](examples/).

### 🧪 Para Pesquisadores e Arquitetos
* **Design Rationale**: Justificativas técnicas e científicas no [Índice de ADRs](docs/adr/INDEX.md).
* **Validação**: Protocolos de integridade descritos na [Especificação FOLD](docs/FOLD_SPECIFICATION.md).

---

## 🛡️ Governança e Qualidade (Caminho Canônico)

Para garantir a integridade sistêmica e evitar a divergência entre plano e código, os artefatos abaixo são as **Fontes da Verdade** na raiz do projeto:

| Eixo | Documentos de Referência |
| :--- | :--- |
| **Planejamento** | [`SUMARIO_COMPLETO.md`](SUMARIO_COMPLETO.md) • [`tarefas_pendentes.md`](tarefas_pendentes.md) |
| **Qualidade/RCA** | [`analise_bugs.md`](analise_bugs.md) • [`revisao_fold_mind.md`](revisao_fold_mind.md) |
| **Implementação** | [`solucoes_fold_mind.py`](solucoes_fold_mind.py) • [`VISUAL_FINAL.txt`](VISUAL_FINAL.txt) |

> **Nota de Sincronização**: Referência atual baseada na branch `work`.

---

## 📈 Validação Local

Para reproduzir os testes de throughput e compressão em seu ambiente:

```bash
# Executa a suíte de benchmarks
python scripts/run_benchmarks.py

# Gera a documentação de performance atualizada
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/BENCHMARKS.md
## Exemplo rápido

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
neuron = MPJRDNeuron(cfg)

x = torch.randn(16, 4, 8)
out = neuron(x, reward=0.2)
print(out["spikes"].shape)
```
