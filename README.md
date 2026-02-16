# PyFolds

## Visão Geral
PyFolds é uma biblioteca em Python/PyTorch para modelagem de neurônios com computação dendrítica (MPJRD), dinâmica de fase (Wave) e serialização `.fold/.mind` com foco em robustez, desempenho e rastreabilidade técnica.

## Instalação Rápida
```bash
pip install pyfolds
```

## Exemplo de Código
```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

config = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
model = MPJRDNeuron(config)

x = torch.randn(32, 4, 8)
out = model(x)
print(out["spikes"])
```

## Mapa de Navegação da Documentação
- **Governança**: `docs/governance/`
  - ADRs: `docs/governance/adr/INDEX.md`
  - Qualidade e auditorias: `docs/governance/quality/`
- **Arquitetura**: `docs/architecture/`
  - Blueprints/UML: `docs/architecture/blueprints/`
  - Especificações: `docs/architecture/specs/`
- **Pesquisa**: `docs/research/`
  - MPJRD: `docs/research/mpjrd/`
  - Wave: `docs/research/wave/`
- **Desenvolvimento**: `docs/development/HUB_CONTROLE.md`
- **Guias públicos**: `docs/public/guides/`
- **Portal da documentação**: `docs/README.md`
