# API Reference - Módulo Core

## MPJRDConfig

**Localização**: `pyfolds.core.config.MPJRDConfig`

### Descrição
Configuração imutável do neurônio MPJRD. Todos os parâmetros são definidos na inicialização e não podem ser alterados durante a execução.

### Parâmetros de Configuração

#### Topologia

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `n_dendrites` | int | 4 | Número de dendritos por neurônio |
| `n_synapses_per_dendrite` | int | 32 | Número de sinapses por dendrito |

#### Filamentos (N) - Memória Estrutural

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `n_min` | int | 0 | Mínimo de filamentos por sinapse |
| `n_max` | int | 31 | Máximo de filamentos por sinapse |
| `w_scale` | float | 5.0 | Fator de escala para peso `W = log2(1+N)/w_scale` |

#### Plasticidade (I) - Memória de Curto Prazo

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `i_eta` | float | 0.01 | Taxa de aprendizado |
| `i_gamma` | float | 0.99 | Fator de decaimento |
| `i_ltp_th` | float | 5.0 | Limiar para LTP (promoção) |
| `i_ltd_th` | float | -5.0 | Limiar para LTD (demoção) |
| `i_min` | float | -20.0 | Valor mínimo de `I` |
| `i_max` | float | 50.0 | Valor máximo de `I` |

#### Homeostase

| Parâmetro | Tipo | Padrão | Descrição |
|---|---|---|---|
| `theta_init` | float | 4.5 | Limiar inicial de disparo |
| `target_spike_rate` | float | 0.1 | Taxa de disparo alvo |
| `homeostasis_alpha` | float | 0.1 | Fator da média móvel |

### Exemplo de Uso

```python
from pyfolds import MPJRDConfig

cfg = MPJRDConfig(
    n_dendrites=8,
    n_synapses_per_dendrite=64,
    target_spike_rate=0.15,
    i_eta=0.02,
)
```

## MPJRDNeuron

**Localização**: `pyfolds.core.neuron.MPJRDNeuron`

### Descrição
Neurônio MPJRD completo com 9 mecanismos biofísicos.

### Métodos Principais

#### `forward(x, reward=None, mode=None, collect_stats=True)`
Processa uma entrada e retorna spikes e métricas.

**Parâmetros:**
- `x` (`torch.Tensor`): Entrada `[batch, dendrites, synapses]`
- `reward` (`float`, opcional): Sinal de recompensa externo
- `mode` (`LearningMode`, opcional): Modo de aprendizado
- `collect_stats` (`bool`): Se deve coletar estatísticas

**Retorno:**

```python
{
    'spikes': torch.Tensor,        # [batch] disparos
    'u': torch.Tensor,             # [batch] potencial somático
    'v_dend': torch.Tensor,        # [batch, dendrites] potenciais dendríticos
    'theta': torch.Tensor,         # Limiar atual
    'r_hat': torch.Tensor,         # Taxa média móvel
    'spike_rate': torch.Tensor,    # Taxa média do batch
    'N_mean': torch.Tensor,        # Média de filamentos
}
```

#### `apply_plasticity(dt=1.0, reward=None)`
Aplica plasticidade baseada em estatísticas acumuladas.

#### `sleep(duration=60.0)`
Executa ciclo de consolidação (sono).

### Exemplo Completo

```python
import torch
from pyfolds import MPJRDConfig, MPJRDNeuron

cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8)
neuron = MPJRDNeuron(cfg)

x = torch.randn(32, 4, 8)
output = neuron(x, reward=0.5)

print(f"Spikes: {output['spikes'].sum().item()}/{32}")
print(f"Taxa média: {output['spike_rate'].item():.3f}")
print(f"Filamentos médios: {output['N_mean'].item():.1f}")

neuron.sleep(duration=100.0)
```
