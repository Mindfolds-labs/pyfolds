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

## Portal de documentação

- 📚 Índice geral: `docs/README.md`
- 🧪 Lógica científica: `docs/SCIENTIFIC_LOGIC.md`
- 🏗️ Arquitetura (C4 + sequência): `docs/ARCHITECTURE.md`
- 🔌 Referência de API: `docs/API_REFERENCE.md`
- 🧭 Guia MNIST: `docs/guides/mnist_example.md`
- 🌊 Tutorial Wave v3.0: `docs/guides/wave_tutorial.md`
- 🤝 Contribuição: `CONTRIBUTING.md`
- 📝 Histórico de versões: `CHANGELOG.md`

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
