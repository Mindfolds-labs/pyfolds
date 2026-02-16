<div align="center">

# PyFolds

[![PyPI](https://img.shields.io/badge/PyPI-pyfolds-blue)](https://pypi.org/project/pyfolds/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-Organized-success)](docs/README.md)

PyFolds é uma biblioteca de alta performance para computação bioinspirada sobre PyTorch, com foco em computação dendrítica não linear, rastreabilidade de estados e integração científica.

</div>

---

## 1. Visão Geral

O framework abstrai a computação dendrítica em um pipeline modular para acelerar pesquisa e engenharia aplicada em modelos MPJRD.

### Por que usar o PyFolds?
- **Modularidade extensível:** componentes com mecanismos de plasticidade e dinâmica de curto prazo.
- **Eficiência nativa:** integração com o ecossistema PyTorch (CPU/GPU).
- **Transparência científica:** separação explícita de sinapse, dendrito e soma para auditoria de estados.

## 2. Instalação

```bash
pip install pyfolds
```

## 3. Quickstart

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

config = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
model = MPJRDNeuron(config)

x = torch.randn(32, 4, 8)  # (batch, dendritos, sinapses)
output = model(x)
print(output["spikes"])
```

## 4. Benchmarks de serialização

```bash
python scripts/run_benchmarks.py --output docs/assets/benchmarks_results.json
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/BENCHMARKS.md
```

Interpretação rápida:
- **Throughput (MiB/s):** quanto maior, melhor.
- **Razão de compressão vs `none`:** valores menores que `1.0` indicam melhor compressão.
- O workflow `.github/workflows/benchmarks.yml` executa periodicamente para atualização de artefatos.

## 5. Portal de documentação

### 5.1 Uso público
- 📑 [Índice de Documentação](docs/README.md)
- 🧪 [Lógica Científica](docs/SCIENTIFIC_LOGIC.md)
- 🏗️ [Arquitetura](docs/ARCHITECTURE.md)
- 📦 [Especificação FOLD](docs/FOLD_SPECIFICATION.md)
- 🔌 [Referência de API](docs/API_REFERENCE.md)
- 📈 [Relatório de Benchmarks](docs/BENCHMARKS.md)

### 5.2 Desenvolvimento e governança (interno)
- 🧭 [Índice Técnico](docs/index.md)
- 🛠️ [Hub de Controle](docs/development/HUB_CONTROLE.md)
- 🧾 [Registro de ADRs](docs/governance/adr/INDEX.md)
- 🛡️ [Plano Mestre de Governança](docs/governance/MASTER_PLAN.md)

## 6. Governança e qualidade (IEEE/ISO)

O processo documental e técnico segue princípios de padronização e rastreabilidade, alinhados a:
- **ISO/IEC 12207** (ciclo de vida de software),
- **IEEE 828** (configuração e controle de mudanças),
- **IEEE 730** (garantia de qualidade).

Referências relevantes no repositório:
- `docs/governance/QUALITY_ASSURANCE.md`
- `docs/governance/RISK_REGISTER.md`
- `docs/governance/adr/INDEX.md`

## 7. Validação local

```bash
python scripts/run_benchmarks.py
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/BENCHMARKS.md
```
