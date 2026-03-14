<div align="center">

<img src="docs/_static/brand/pyfolds-readme-banner.svg" alt="PyFolds banner" width="100%" />

<img src="src/pyfolds/assets/icons/pyfolds.svg" alt="PyFolds icon" width="112" />

# PyFolds

[![PyPI](https://img.shields.io/badge/PyPI-pyfolds-blue)](https://pypi.org/project/pyfolds/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)
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

### Requisitos mínimos (alinhados ao `pyproject.toml`)

- Python `>=3.8`
- PyTorch `>=2.0`
- numpy `>=1.24`
- zstandard `>=0.21`
- google-crc32c `>=1.5`
- reedsolo `>=1.7`

### Extras opcionais

- `pip install pyfolds[full]` para stack completa (serialização, telemetria e utilitários extras).
- `pip install pyfolds[serialization]` para codecs/formatos de serialização.
- `pip install pyfolds[telemetry]` para visualização de métricas.
- `pip install pyfolds[dev]` para lint/testes.
- `pip install pyfolds[examples]` para dependências de exemplos.
- `pip install pyfolds[tensorflow]` para backend TensorFlow (`tensorflow-cpu>=2.20.0`).


## Brand oficial

- Especificação do ícone: `docs/brand/ICON_SPEC.md`.
- Asset canônico de documentação: `docs/_static/brand/`.
- Asset para uso no pacote (quando necessário): `src/pyfolds/assets/brand/`.
- Pipeline reprodutível de geração:

```bash
python docs/brand/render_assets.py
```

## Documentation

- Instale dependências de documentação:

```bash
pip install -r requirements-docs.txt
```

- Gere o site localmente com Sphinx:

```bash
sphinx-build -b html docs docs/_build/html
```

- Entrada principal da documentação: `docs/index.md`.
- Portal HUB já existente: `docs/development/HUB_CONTROLE.md` (use como base, sem recriação do zero).
- Publicação (quando disponível no CI/GitHub Pages): consultar workflow de docs do repositório.

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

## 4. Política de depreciação da API pública

Os aliases legados da v1 continuam disponíveis na série `2.x`, emitindo `DeprecationWarning` para facilitar migração gradual sem quebra imediata.

- `MPJRDConfig` → `NeuronConfig`
- `MPJRDLayer` → `AdaptiveNeuronLayer`
- `MPJRDNetwork` → `SpikingNetwork`

Critérios objetivos adotados:
- manutenção por ciclo mínimo de uma major completa (`2.x`),
- aviso explícito no `CHANGELOG.md`,
- estratégia de migração com mapeamento 1:1 para nomes canônicos v2.

Versão-limite planejada: remoção dos aliases v1 em uma próxima major (`3.0.0`), após aviso explícito em changelog.

## 5. Benchmarks de serialização

```bash
python scripts/run_benchmarks.py --output docs/assets/benchmarks_results.json
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/assets/BENCHMARKS.md
```

Interpretação rápida:
- **Throughput (MiB/s):** quanto maior, melhor.
- **Razão de compressão vs `none`:** valores menores que `1.0` indicam melhor compressão.
- O workflow `.github/workflows/benchmarks.yml` executa periodicamente para atualização de artefatos.

## 6. Portal de documentação

### 5.1 Uso público
- 📑 [Índice de Documentação](docs/README.md)
- 🧪 [Lógica Científica](docs/science/SCIENTIFIC_LOGIC.md)
- 🏗️ [Arquitetura](docs/ARCHITECTURE.md)
- 📦 [Especificação FOLD](docs/architecture/specs/FOLD_SPECIFICATION.md)
- 🔌 [Referência de API](docs/api/API_REFERENCE.md)
- 📈 [Relatório de Benchmarks](docs/assets/BENCHMARKS.md)

### 5.2 Desenvolvimento e governança (interno)
- 🧭 [Índice Técnico](docs/index.md)
- 🛠️ [Hub de Controle](docs/development/HUB_CONTROLE.md)
- 🧾 [Registro de ADRs](docs/governance/adr/INDEX.md)
- 🛡️ [Plano Mestre de Governança](docs/governance/MASTER_PLAN.md)

## 7. Governança e qualidade (IEEE/ISO)

O processo documental e técnico segue princípios de padronização e rastreabilidade, alinhados a:
- **ISO/IEC 12207** (ciclo de vida de software),
- **IEEE 828** (configuração e controle de mudanças),
- **IEEE 730** (garantia de qualidade).

Referências relevantes no repositório:
- [QUALITY_ASSURANCE.md](./docs/governance/QUALITY_ASSURANCE.md)
- [RISK_REGISTER.md](./docs/governance/RISK_REGISTER.md)
- [adr/INDEX.md](./docs/governance/adr/INDEX.md)
- [analise_bugs.md](./docs/governance/analise_bugs.md)
- [revisao_fold_mind.md](./docs/governance/revisao_fold_mind.md)
- [tarefas_pendentes.md](./docs/governance/tarefas_pendentes.md)
- [solucoes_fold_mind.py](./docs/governance/solucoes_fold_mind.py)
- [VISUAL_FINAL.txt](./docs/governance/VISUAL_FINAL.txt)

## 8. Validação local

```bash
python scripts/run_benchmarks.py
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/assets/BENCHMARKS.md
```

## 9. Workflow v6 (CRIAR → ANALISAR → EXECUTAR → FINALIZAR)

Fluxo operacional canônico para governança e execução de issues:

1. **CRIAR**
   - Criar relatório em `docs/development/prompts/relatorios/ISSUE-XXX-slug.md`.
   - Criar log em `docs/development/prompts/logs/ISSUE-XXX-slug-LOG.md`.
2. **ANALISAR**
   - Validar escopo, riscos, critérios de aceite e artefatos afetados.
3. **EXECUTAR**
   - Implementar mudanças e manter rastreabilidade em `docs/development/execution_queue.csv`.
4. **FINALIZAR**
   - Sincronizar HUB, rodar validações e abrir PR com evidências.

Comandos mínimos:

```bash
python -m compileall src/
python tools/validate_docs_links.py
python tools/sync_hub.py --check
PYTHONPATH=src pytest tests/ -v
```

## 10. Links importantes (desenvolvimento)

- 📁 [Portal de Prompts Operacionais](docs/development/prompts/README.md)
- 🧾 [Relatórios canônicos](docs/development/prompts/relatorios/)
- 🗂️ [Logs canônicos](docs/development/prompts/logs/)
- 🛠️ [HUB de Controle](docs/development/HUB_CONTROLE.md)
- 📌 [Fila de Execução](docs/development/execution_queue.csv)
