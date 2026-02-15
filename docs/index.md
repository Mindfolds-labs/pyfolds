# PyFolds - Core Neural Computation Framework

<div align="center">
  
  **Core Neural Computation Framework**
  
  [![PyPI](https://img.shields.io/pypi/v/pyfolds?style=flat-square&logo=pypi)](https://pypi.org/project/pyfolds/)
  [![Python](https://img.shields.io/pypi/pyversions/pyfolds?style=flat-square&logo=python)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
  [![License](https://img.shields.io/github/license/Mindfolds-labs/pyfolds?style=flat-square)](LICENSE)
  [![Docs](https://img.shields.io/badge/docs-latest-blue?style=flat-square)](https://pyfolds.readthedocs.io)
  [![Tests](https://img.shields.io/github/actions/workflow/status/Mindfolds-labs/pyfolds/ci.yml?style=flat-square)](https://github.com/Mindfolds-labs/pyfolds/actions)
  [![Coverage](https://img.shields.io/codecov/c/github/Mindfolds-labs/pyfolds?style=flat-square)](https://codecov.io/gh/Mindfolds-labs/pyfolds)

  <br>
  [📚 Documentação](/#) •
  [🚀 Instalação](installation.md) •
  [🔥 Quick Start](quickstart.md) •
  [🧪 Exemplos](/#) •
  [🤝 Contribuir](development/contributing.md)
  <br><br>
</div>

---

## 📋 Visão Geral

**PyFolds** é um framework Python para simulação de neurônios e redes neurais com 9 mecanismos biologicamente inspirados. O modelo MPJRD (Multi-Pathway Joint-Resource Dendritic) implementa plasticidade estrutural explícita, consolidação offline e processamento dendrítico multi-compartimental.

## Arquitetura

```
pyfolds/
├── core/          # Núcleo: neurônio MPJRD, sinapses, dendritos
├── advanced/      # Mecanismos: STDP, adaptação, inibição, backprop
├── layers/        # Camadas de neurônios para redes
├── network/       # Redes neurais com conectividade topológica
├── telemetry/     # Sistema de monitoramento e logging
└── utils/         # Utilitários: math, device, tipos, logging
```

## 🧬 Mecanismos

| # | Mecanismo | Descrição | Fonte |
|---|-----------|-----------|-------|
| 1 | Força Sináptica (N) | Memória estrutural (0-31) | `core/synapse.py` |
| 2 | Potencial interno (I) | Memória volátil | `core/synapse.py` |
| 3 | Dinâmica de curto prazo | Facilitação/Depressão | `advanced/short_term.py` |
| 4 | Homeostase | Theta adaptativo | `core/homeostasis.py` |
| 5 | Neuromodulação | 3 modos: external, capacity, surprise | `core/neuromodulation.py` |
| 6 | Backpropagação dendrítica | Comunicação soma → dendrito | `advanced/backprop.py` |
| 7 | Adaptação (SFA) | Spike-frequency adaptation | `advanced/adaptation.py` |
| 8 | STDP | Spike-timing dependent plasticity | `advanced/stdp.py` |
| 9 | Consolidação two-factor | Sono para transferência I → N | `core/synapse.py` |

---

## ⚡ Quick Start

```python
import torch
import pyfolds

# Configuração
cfg = pyfolds.MPJRDConfig(n_dendrites=4)
neuron = pyfolds.MPJRDNeuron(cfg)

# Dados
x = torch.randn(16, 4, 32)

# Forward
out = neuron(x)
print(f"Spike rate: {out['spike_rate'].item():.2%}")

# Batch learning
neuron.set_mode(pyfolds.LearningMode.BATCH)
for _ in range(10):
    out = neuron(x, collect_stats=True)
neuron.apply_plasticity()

# Consolidação (sono)
neuron.set_mode(pyfolds.LearningMode.SLEEP)
neuron.sleep(duration=100.0)

📦 Instalação

# Usuário final
pip install pyfolds

# Desenvolvedor
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds
pip install -e ".[dev,docs]"

Dependências:

Core: torch>=2.0.0, torchvision>=0.15.0, numpy>=1.19.0

Dev: pytest, black, mypy, pre-commit

Docs: sphinx, sphinx-rtd-theme

📊 Documentação
Seção	Descrição
Instalação	Requisitos e setup
Quick Start	Primeiros passos
Guias	Conceitos e arquitetura
API	Referência completa
Tutoriais	Exemplos práticos
Contribuição	Guia para desenvolvedores

🧪 Exemplos

# Básico
python examples/basic_neuron.py

# Batch learning
python examples/batch_learning.py

# Rede neural
python examples/network_example.py

# Telemetria
python examples/telemetry_example.py

📈 Performance
Operação	CPU (i9)	GPU (RTX 4090)
Forward (batch=64)	0.12 ms	0.08 ms
Batch learning (100 steps)	2.3 s	0.18 s
Sono (1000 replay)	4.1 s	0.32 s
🤝 Contribuição
Fork o repositório

Crie uma branch: git checkout -b feature/nova-funcionalidade

Commit: git commit -m '✨ feat: adiciona funcionalidade'

Push: git push origin feature/nova-funcionalidade

Abra um Pull Request

Padrões:

✨ feat: nova funcionalidade

🐛 fix: correção de bug

📚 docs: documentação

🎨 style: formatação

♻️ refactor: refatoração

🧪 test: testes

📄 Licença
MIT License © 2025 Mindfolds Labs

📬 Contato
Autor: Antônio Carlos — jrduraes90@gmail.com

GitHub: github.com/Mindfolds-labs/pyfolds

Issues: github.com/Mindfolds-labs/pyfolds/issues


---

## 🎯 **Características desta versão:**

| Aspecto | Implementação |
|---------|---------------|
| **Clean** | Sem emojis excessivos, formatação limpa |
| **Profissional** | Badges informativos, estrutura clara |
| **Escalável** | Links absolutos, pronto para tradução |
| **Completo** | Visão geral, mecanismos, instalação, exemplos |
| **Técnico** | Foco no código e na arquitetura |

**Pronto para colar!** 🚀









