<div align="center">

# 🧠 PyFolds v2.0/v3.0

[![PyPI](https://img.shields.io/badge/PyPI-pyfolds-blue)](https://pypi.org/project/pyfolds/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Framework neurocomputacional bioinspirado com computação dendrítica não-linear, plasticidade estrutural e consolidação offline.**

</div>

---

## Resumo Executivo

O **PyFolds** implementa o modelo **MPJRD (Multi-Pathway Joint-Resource Dendritic)** para substituir parte do papel das camadas ocultas por uma dinâmica explícita de **Sinapse → Dendrito → Soma → Axônio (onda/fase)**. Em vez de uma “caixa preta” de ativações internas difíceis de interpretar, o sistema expõe estados fisiologicamente inspirados (`N`, `I`, `W`, `theta`, `R`) em cada etapa de decisão e aprendizado.

### Por que isso reduz o problema da caixa-preta?

- O estado de memória de longo prazo é explícito em `N` (filamentos discretos por sinapse).
- A integração de evidências é observável em `v_dend` (por dendrito).
- A decisão somática é auditável por `u`, `theta` e `spikes`.
- A consolidação ("sono") separa aquisição online de estabilização offline.

---

## Visão Geral do MPJRD

```mermaid
flowchart LR
    X[Entrada x\n[B, D, S]] --> S[Sinapse\nEstado: N, I, W]
    S --> D[Dendrito\nSubunidade não-linear]
    D --> P{Processamento paralelo\npor D dendritos}
    P --> SOMA[Soma\nIntegração cooperativa]
    SOMA --> AX[Axônio\nSpike / Onda-Fase]
    AX --> PL[Plasticidade + Consolidação]
```

### Hipóteses centrais

1. **Quantização estrutural (`N`)**: memória robusta e interpretável por estados discretos.
2. **Subunidades dendríticas**: computação local não-linear antes da decisão global.
3. **Integração somática cooperativa**: evita colapso informacional típico de seleção dura de um único caminho.
4. **Aprendizado em duas escalas**: atualização online + consolidação offline.

---

## Quick Start

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(
    n_dendrites=4,
    n_synapses_per_dendrite=8,
    plastic=True,
)

neuron = MPJRDNeuron(cfg)
x = torch.randn(32, 4, 8)
out = neuron(x, reward=0.25)

print(out["spikes"].shape)    # [32]
print(out["v_dend"].shape)    # [32, 4]
print(out["N_mean"].item())   # Estado estrutural médio
```

---

## Instalação

### Via pip

```bash
pip install pyfolds
```

### Desenvolvimento local

```bash
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Guia de Leitura da Documentação (C4 + Ciência)

- `docs/SCIENTIFIC_LOGIC.md` → fundamento científico e formalismo.
- `docs/ARCHITECTURE.md` → desenho de sistema em camadas (C4).
- `docs/ALGORITHM.md` → passo a passo do forward e consolidação offline.
- `docs/API_REFERENCE.md` → API técnica das classes centrais.

---

## Roadmap da Documentação v2.0/v3.0

- [x] Estrutura executiva do README.
- [x] Núcleo teórico inicial (`SCIENTIFIC_LOGIC`).
- [ ] Arquitetura detalhada com transição Hard-WTA → Integração Cooperativa.
- [ ] Algoritmo matemático completo (forward + sono).
- [ ] Referência de API consolidada.
