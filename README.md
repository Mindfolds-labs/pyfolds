<div align="center">
  
  # 🧠 PyFOLDS
  
  [![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/pyfolds/)
  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![Imports: isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://pycqa.github.io/isort/)
  [![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)
  
  **PyFolds - Core Neural Computation Frameworks**
  
  *Neurociência computacional biofisicamente plausível com plasticidade estrutural e consolidação offline*
  
  [📚 Documentação](#-documentação) • 
  [🚀 Instalação](#-instalação) • 
  [🔥 Quick Start](#-quick-start) • 
  [🧪 Exemplos](#-exemplos) • 
  [🤝 Contribuir](#-contribuindo)
  
  ---
  
  **Author:** Antônio Carlos ([jrduraes90@gmail.com](mailto:jrduraes90@gmail.com))
  
</div>

---

## 📋 Tabela de Conteúdos

- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Quick Start](#-quick-start)
- [Componentes](#-componentes)
- [Exemplos](#-exemplos)
- [Documentação](#-documentação)
- [Performance](#-performance)
- [Contribuição](#-contribuindo)
- [Licença](#-licença)
- [Citação](#-citação)
- [Contato](#-contato)

---

## 🧠 Visão Geral

**PyFOLDS** (Framework for Organizing Learning and Dendritic Structures) é um framework moderno de neurociência computacional que implementa o modelo **MPJRD (Multi-Pathway Joint-Resource Dendritic)** - um neurônio estrutural com plasticidade sináptica explícita, consolidação offline e atenção espacial.

### ✨ Diferenciais

| Característica | PyFOLDS | Frameworks Tradicionais |
|---------------|---------|------------------------|
| **Dendritos explícitos** | ✅ [B, D, S] | ❌ Apenas [B, N] |
| **Plasticidade estrutural** | ✅ Níveis N + pesos W | ❌ Apenas pesos |
| **Sinapses com proteção** | ✅ Estado de saturação | ❌ Não existe |
| **Consolidação offline** | ✅ Replay + sono + meta | ❌ Apenas online |
| **Atenção espacial** | ✅ Ganho topográfico | ❌ Não existe |
| **Homeostase adaptativa** | ✅ Limiar θ dinâmico | ⚠️ Limitado |

### 🎯 Aplicações

- 🧬 **Modelagem neurocientífica** - Córtex visual, hipocampo, plasticidade
- 🤖 **IA bioinspirada** - Redes neurais com aprendizagem contínua
- 🧠 **Memória e consolidação** - Replay, sono, metaplasticidade
- 👁️ **Atenção visual** - Foco espacial, ganho sináptico

---

## 🏗️ Arquitetura

```text
┌─────────────────────────────────────────────────────────────┐
│ PYFOLDS                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────┐     │
│  │      CORE       │    │         NETWORKS            │     │
│  │   (Imutável)    │    │       (Extensível)          │     │
│  ├─────────────────┤    ├─────────────────────────────┤     │
│  │ • Config        │    │ • MPJRDNetwork              │     │
│  │ • Synapse (GLW) │    │ • Projections               │     │
│  │ • Dendrite      │    │ • ActivityBuffer            │     │
│  │ • Neuron MPJRD  │    │ • ConsolidationModule       │     │
│  │ • Factory       │    │ • SpatialAttention          │     │
│  └─────────────────┘    └─────────────────────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │            PLASTICIDADE ESTRUTURAL                   │     │
│  ├─────────────────────────────────────────────────────┤     │
│  │  Nível N │  Peso W  │ Corrente I │   Modo Proteção   │     │
│  │  ┌───┐     ┌───┐      ┌───┐       ┌───┐             │     │
│  │  │0-31│  → │log2│   → │ I │  LTP  │🛡️ │             │     │
│  │  └───┘     └───┘      └───┘   →   └───┘             │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

pyfolds/
├── pyfolds/
│   ├── __init__.py           # API pública
│   │
│   ├── core/                 # Núcleo do neurônio MPJRD
│   │   ├── __init__.py
│   │   ├── config.py         # MPJRDConfig (dataclass)
│   │   ├── synapse.py        # Sinapse GLW com proteção
│   │   ├── dendrite.py       # Dendrito com N sinapses
│   │   ├── neuron.py         # Neurônio MPJRD completo
│   │   └── factory.py        # build_mpjrd()
│   │
│   ├── networks/             # Redes multicamadas
│   │   ├── __init__.py
│   │   ├── network.py        # MPJRDNetwork, projeções
│   │   ├── buffer.py         # ActivityBuffer (replay)
│   │   ├── consolidation.py  # ConsolidationModule (sono)
│   │   └── attention.py      # Atenção espacial
│   │
│   └── scripts/              # Utilitários
│       ├── __init__.py
│       └── info.py           # pyfolds-info
│
├── tests/                    # Testes unitários
├── examples/                 # Exemplos completos
├── docs/                     # Documentação
│
├── setup.cfg                 # Configuração do pacote
├── pyproject.toml            # Ferramentas (black, mypy)
├── Makefile                  # Automação completa
├── requirements.txt          # Dependências
└── README.md                 # Você está aqui


🚀 Instalação
📋 Pré-requisitos
Python 3.8 ou superior

pip (gerenciador de pacotes)

[Opcional] CUDA para GPU

⚡ Método 1: Instalação Automática (Recomendado)
bash

# Clone o repositório
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds

# Instalação com Make (CPU)
make install

# OU para GPU (CUDA 11.8)
make install-cuda

# Verifique a instalação
pyfolds-info

# Clone o repositório
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds

# Instalação com Make (CPU)
make install

# OU para GPU (CUDA 11.8)
make install-cuda

# Verifique a instalação
pyfolds-info

🛠️ Método 3: Ambiente de Desenvolvimento
bash
# Ambiente virtual completo com ferramentas de dev
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

make install-dev

✅ Verificação da Instalação
python
python -c "from pyfolds import build_mpjrd; print('✅ PyFOLDS instalado com sucesso!')"
Ou execute:

bash
pyfolds-info
Saída esperada:

text
==================================================
🔬 PyFOLDS - System Information
==================================================
📦 PyFOLDS:      v0.1.0
🐍 Python:        3.10.12
🔥 PyTorch:       2.1.0
   CUDA:          True
   Device:        NVIDIA GeForce RTX 4090
📊 NumPy:         1.24.3
💻 Sistema:       Linux 6.2.0
==================================================
✅ PyFOLDS instalado corretamente!
🔥 Quick Start
1️⃣ Neurônio Único

from pyfolds import build_mpjrd
import torch

# Cria neurônio com 4 dendritos, 8 sinapses cada
neuron = build_mpjrd(
    n_dendrites=4, 
    n_synapses_per_dendrite=8,
    seed=42  # Reprodutibilidade
)

# Entrada: [batch, dendritos, sinapses]
x = torch.randn(32, 4, 8)

# Forward pass com plasticidade
out = neuron.step(x, reward=0.5, dt=1.0)

print(f"🔹 Spikes: {out['spikes'].shape}")        # [32]
print(f"🔹 Taxa média: {out['spike_rate']:.3f}")  # 0.125
print(f"🔹 Limiar (θ): {out['theta'].item():.3f}")# 4.500
print(f"🔹 Saturação: {out['saturation_ratio']:.1%}")  # 3.2%

2️⃣ Rede Multicamadas V1 → V2 → V3

from pyfolds import MPJRDConfig
from pyfolds.networks import build_v1_v2_v3_network

# Configurações por camada
cfg_v1 = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
cfg_v2 = MPJRDConfig(n_dendrites=6, n_synapses_per_dendrite=12)
cfg_v3 = MPJRDConfig(n_dendrites=8, n_synapses_per_dendrite=16)

# Rede hierárquica topográfica
net = build_v1_v2_v3_network(
    cfg_v1, cfg_v2, cfg_v3,
    n_v1=64,    # Grid 8x8
    n_v2=100,   # Posições aleatórias
    n_v3=50,    # Posições aleatórias
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Entrada para V1: [batch, neurônios, dendritos, sinapses]
inputs = {
    'V1': torch.randn(32, 64, 4, 8, device=net.device)
}

# Forward
outputs = net(inputs, reward=0.3, dt=1.0)

print(f"🔹 Taxa V1: {outputs['V1']['spike_rate'].mean():.3f}")
print(f"🔹 Taxa V2: {outputs['V2']['spike_rate'].mean():.3f}") 
print(f"🔹 Taxa V3: {outputs['V3']['spike_rate'].mean():.3f}")

3️⃣ Consolidação Offline (Sono)

from pyfolds.networks import ActivityBuffer, ConsolidationModule

# Buffer de experiências (capacidade 10k)
buffer = ActivityBuffer(maxlen=10000)

# Fase de aquisição (online)
for episode in range(100):
    inputs = {'V1': torch.randn(16, 64, 4, 8)}
    outputs = net(inputs, reward=0.5, dt=1.0)
    
    # Armazena atividade
    buffer.add(
        step=episode,
        activations=outputs,
        positions=net.positions,
        context={'task': 'visual', 'episode': episode}
    )

# Consolidação offline (sono)
consolidation = ConsolidationModule(
    network=net,
    buffer=buffer,
    lr_offline=0.001,  # Taxa de aprendizado mais baixa
    replay_batch=32
)

# Ciclo de sono: replay + homeostase + metaplasticidade
consolidation.sleep_cycle(n_replay=5)

print(f"💤 Consolidação concluída: {len(buffer)} experiências replay")

4️⃣ Atenção Espacial

from pyfolds.networks import SpatialAttention

# Posições dos neurônios (coordenadas 2D)
src_pos = net.positions['V1']  # [64, 2]
dst_pos = net.positions['V2']  # [100, 2]

# Módulo de atenção espacial
attention = SpatialAttention(
    src_pos=src_pos,
    dst_pos=dst_pos,
    D=cfg_v2.n_dendrites,
    S=cfg_v2.n_synapses_per_dendrite,
    sigma=0.2,      # Largura do foco
    amplitude=2.0   # Ganho máximo
)

# Foco atencional no centro do grid
focus = torch.tensor([[0.5, 0.5]])  # [B, 2]

# Ganho sináptico baseado na distância
gain = attention(focus)  # [1, 64, 100, 6, 12]

print(f"🎯 Ganho máximo: {gain.max().item():.3f}")
print(f"🎯 Ganho médio: {gain.mean().item():.3f}")
🧩 Componentes
🧬 Core (Núcleo)
Classe	Descrição	Parâmetros Chave
MPJRDConfig	Configuração do neurônio	n_dendrites, n_synapses, i_eta
MPJRDSynapse	Sinapse GLW com proteção	N, W, I, protection_mode
MPJRDDendrite	Dendrito com N sinapses	synapses, forward()
MPJRDNeuron	Neurônio completo	dendrites, theta, step()
build_mpjrd()	Factory function	**kwargs, device, seed
🔌 Networks (Redes)
Classe	Descrição	Métodos Principais
MPJRDNetwork	Rede multicamadas	add_population(), add_connection()
MPJRDProjection	Projeção sináptica	forward(src_spikes)
ActivityBuffer	Buffer de replay	add(), sample(), clear()
ConsolidationModule	Sono e metaplasticidade	sleep_cycle(), replay_hebbian()
SpatialAttention	Atenção topográfica	forward(focus)
📐 Funções de Conectividade
python
# Conexão Gaussiana (topográfica)
gaussian_connectivity(
    src_pos, dst_pos, D, S, 
    sigma=0.2, amplitude=1.0, normalize=True
)

# Conexão 1-1 (vizinho mais próximo)
one_to_one_connectivity(
    src_pos, dst_pos, D, S, 
    strength=1.0
)
💡 Exemplos
📁 examples/basic_neuron.py
python
"""Neurônio isolado com diferentes neuromodulações."""
from pyfolds import build_mpjrd, MPJRDConfig

# Modo: neuromodulação externa (reward)
cfg_ext = MPJRDConfig(neuromod_mode="external")
neuron_ext = build_mpjrd(config=cfg_ext)

# Modo: neuromodulação por capacidade
cfg_cap = MPJRDConfig(neuromod_mode="capacity", cap_k_sat=1.5)
neuron_cap = build_mpjrd(config=cfg_cap)

# Modo: neuromodulação por surpresa
cfg_sup = MPJRDConfig(neuromod_mode="surprise", sup_k=3.0)
neuron_sup = build_mpjrd(config=cfg_sup)
📁 examples/visual_hierarchy.py
python
"""Rede V1→V2→V3 com conectividade topográfica."""
from pyfolds.networks import build_v1_v2_v3_network, SpatialAttention

# Build network
net = build_v1_v2_v3_network(...)

# Adiciona atenção entre V1 e V2
attention = SpatialAttention(
    net.positions['V1'], 
    net.positions['V2'],
    D=6, S=12
)

# Aplica ganho atencional durante forward
focus = get_focus_position()  # [B, 2]
gain = attention(focus)
📁 examples/consolidation_demo.py
python
"""Demo completa de consolidação offline."""
# 1. Treino online
for step in range(1000):
    out = net(inputs)
    buffer.add(step, out, net.positions)

# 2. Sono profundo
consolidation.sleep_cycle(n_replay=10)

# 3. Avaliação pós-consolidação
test_outputs = net(test_inputs)
📁 examples/attention_navigation.py
python
"""Atenção espacial dinâmica para navegação."""
# Foco segue alvo móvel
for t in range(100):
    focus = target_position[t]  # [1, 2]
    gain = attention(focus)
    
    # Aplica ganho na projeção
    proj.weights = proj.weights * gain
📚 Documentação
📖 Guias
Guia	Descrição	Link
Arquitetura	Visão detalhada do design	ARCHITECTURE.md
API Reference	Documentação completa das classes	API.md
Guia de Uso	Tutoriais passo a passo	guides/USAGE.md
Fluxos	Diagramas de execução	FLOWS.md
ADRs	Decisões arquiteturais	adr/
🎓 Tutoriais Rápidos
bash
# Tutorial 1: Primeiros passos
python examples/basic_neuron.py

# Tutorial 2: Rede visual
python examples/visual_hierarchy.py

# Tutorial 3: Consolidação
python examples/consolidation_demo.py

# Tutorial 4: Atenção
python examples/attention_navigation.py
🔧 Configuração Avançada
python
from pyfolds import MPJRDConfig

# Configuração completa
cfg = MPJRDConfig(
    # Arquitetura
    n_dendrites=8,
    n_synapses_per_dendrite=32,
    
    # Plasticidade
    i_eta=0.01,
    i_gamma=0.99,
    beta_w=0.1,
    
    # Limites
    n_min=0,
    n_max=31,
    i_min=-20.0,
    i_max=50.0,
    
    # Homeostase
    theta_init=4.5,
    theta_min=2.0,
    theta_max=8.0,
    target_spike_rate=0.1,
    
    # Neuromodulação
    neuromod_mode="capacity",
    cap_k_sat=1.2,
    cap_k_rate=0.8
)
⚡ Performance
📊 Benchmarks
Operação	CPU (i9)	GPU (RTX 4090)	Speedup
Neurônio único (forward)	0.12 ms	0.08 ms	1.5x
Rede V1-V2-V3 (100 steps)	2.3 s	0.18 s	12.8x
Consolidação (1000 replay)	4.1 s	0.32 s	12.8x
Atenção espacial	0.05 ms	0.03 ms	1.7x
🚀 Otimizações
python
# Ativar modo eval (desativa gradientes)
net.eval()

# Batch processing
batch_size = 128  # Aumente conforme GPU

# Mixed precision (se disponível)
with torch.cuda.amp.autocast():
    outputs = net(inputs)
🤝 Contribuindo
📋 Diretrizes
Fork o repositório

Crie uma branch (git checkout -b feature/nova-funcionalidade)

Commit suas mudanças (git commit -m '✨ feat: adiciona nova funcionalidade')

Push para a branch (git push origin feature/nova-funcionalidade)

Abra um Pull Request

🔧 Setup de Desenvolvimento
bash
# Clone e instale com dependências dev
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds
make install-dev

# Rode os testes
make test

# Verifique o estilo do código
make lint

# Formate o código
make format
✅ Padrões de Commit
Tipo	Descrição	Exemplo
✨ feat	Nova funcionalidade	✨ feat: adiciona atenção multi-foco
🐛 fix	Correção de bug	🐛 fix: corrige deadlock no replay
📚 docs	Documentação	📚 docs: atualiza API reference
🎨 style	Formatação	🎨 style: aplica black/isort
♻️ refactor	Refatoração	♻️ refactor: otimiza update sináptico
🧪 test	Testes	🧪 test: adiciona test para saturação
⚡ perf	Performance	⚡ perf: acelera forward com einsum