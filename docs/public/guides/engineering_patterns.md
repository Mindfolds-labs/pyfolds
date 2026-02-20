# Guia de Engenharia: Factory, Validação, Checkpoint e Health Check

Este guia mostra **como usar na prática** os blocos de engenharia adicionados no PyFolds para tornar o treinamento mais seguro, reproduzível e observável.

---

## 1) Factory de neurônios (baixo acoplamento)

Use `create_neuron` para deixar a escolha de classe desacoplada do código cliente.

```python
from pyfolds.core import create_neuron
from pyfolds.core import MPJRDConfig
from pyfolds.wave import MPJRDWaveConfig

core_neuron = create_neuron(MPJRDConfig())
wave_neuron = create_neuron(MPJRDWaveConfig())
```

### Quando isso ajuda?
- cenários com múltiplos tipos de neurônio;
- pipelines de experimento guiados por configuração;
- extensão futura com novos tipos registrados no `NeuronFactory`.

---

## 2) Validação de entrada no `forward`

Os neurônios validam automaticamente:
- número de dimensões;
- shape esperado (`[batch, dendrites, synapses]`);
- tipo numérico de ponto flutuante.

### Exemplo de erro esperado

```python
import torch
import pyfolds

cfg = pyfolds.MPJRDConfig()
neuron = pyfolds.MPJRDNeuron(cfg)

x_invalido = torch.randint(0, 2, (2, cfg.n_dendrites, cfg.n_synapses_per_dendrite))
neuron(x_invalido)  # TypeError: x deve ser tensor de ponto flutuante
```

---

## 3) Checkpoint versionado com integridade

```python
import pyfolds
from pyfolds.serialization import VersionedCheckpoint

cfg = pyfolds.MPJRDConfig()
model = pyfolds.MPJRDNeuron(cfg)

ckpt = VersionedCheckpoint(model, version="2.0.3")
ckpt.save("artifacts/neuron.pt", extra_metadata={"run": "exp-42"})

loaded = VersionedCheckpoint.load("artifacts/neuron.pt", model=model)
print(loaded["metadata"]["git_hash"])  # hash de commit (quando disponível)
```

### O que é persistido
| Campo | Objetivo |
|---|---|
| `model_state` | estado completo do modelo |
| `metadata.version` | versão lógica do artefato |
| `metadata.git_hash` | rastreabilidade de código |
| `metadata.config` | parâmetros do modelo |
| `integrity_hash` | verificação de integridade |

---

## 4) Health check operacional

```python
from pyfolds.monitoring import NeuronHealthCheck

checker = NeuronHealthCheck(model)
status, alerts = checker.check()
print(status.value, alerts)
```

### Semântica de status
- `healthy`: sem sinais de degradação;
- `degraded`: saturação alta ou taxa de disparo muito baixa;
- `critical`: taxa alta de neurônios mortos.

---

## 5) Boas práticas rápidas

- Centralize criação via factory em scripts de treino.
- Rode health check por época e logue alertas.
- Sempre salve checkpoints com metadados de experimento.
- Em pipelines, valide lote de entrada antes de benchmark de performance.
