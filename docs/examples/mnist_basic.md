# Tutorial: Classificação MNIST com PyFolds

Este tutorial mostra como usar neurônios MPJRD para classificar dígitos MNIST sem camadas ocultas.

## 1. Configuração Inicial

```python
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from pyfolds import MPJRDConfig, MPJRDNeuron
from pyfolds.utils.types import LearningMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")
```

## 2. Carregar Dados MNIST

```python
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    torchvision.transforms.Lambda(lambda x: x.view(-1)),
])

train_dataset = torchvision.datasets.MNIST(
    './data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    './data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

## 3. Configurar Neurônios para MNIST

```python
class MNISTNetwork(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.cfg = MPJRDConfig(
            n_dendrites=7,
            n_synapses_per_dendrite=112,
            target_spike_rate=0.1,
            i_eta=0.01,
            theta_init=4.5,
        )
        self.neurons = nn.ModuleList([MPJRDNeuron(self.cfg) for _ in range(n_classes)])
        self.receptive_fields = self._create_receptive_fields()

    def _create_receptive_fields(self):
        fields = []
        for i in range(7):
            mask = torch.zeros(784)
            mask[i * 112:(i + 1) * 112] = 1.0
            fields.append(mask)
        return torch.stack(fields)

    def forward(self, x, reward=None, mode=LearningMode.ONLINE):
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(1) * self.receptive_fields.to(x.device).unsqueeze(0)
        x_dendrites = x_expanded.view(batch_size, 7, -1)

        outputs = []
        for neuron in self.neurons:
            out = neuron(x_dendrites, reward=reward, mode=mode)
            outputs.append(out)

        phases = torch.stack([out.get('phase', out['spike_rate']) for out in outputs])
        predictions = phases.argmin(dim=0)
        return predictions, outputs
```

## 4. Treinamento

```python
def train(model, train_loader, epochs=5):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            predictions, outputs = model(data, mode=LearningMode.ONLINE)

            for i, (pred, out) in enumerate(zip(predictions, outputs)):
                reward = 1.0 if pred == target else -0.5
                model.neurons[i].apply_plasticity(reward=reward)

            correct += (predictions == target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                acc = 100.0 * correct / total
                print(f'Epoch {epoch}, Batch {batch_idx}, Acc: {acc:.2f}%')

        for neuron in model.neurons:
            neuron.sleep(duration=50.0)

        acc = 100.0 * correct / total
        print(f'Epoch {epoch} completa. Acurácia: {acc:.2f}%')
```

## 5. Resultados Esperados

- Acurácia: ~92-95% após 5 épocas.
- Neurônios especializados por classe.
- Filamentos médios: 15-25 por sinapse ativa.

- > ## 2. Carregar Dados MNIST
>
> ```python
> transform = torchvision.transforms.Compose([
>     torchvision.transforms.ToTensor(),
>     torchvision.transforms.Normalize((0.1307,), (0.3081,)),
>     torchvision.transforms.Lambda(lambda x: x.view(-1)),
> ])
>
> train_dataset = torchvision.datasets.MNIST(
>     './data', train=True, download=True, transform=transform
> )
> test_dataset = torchvision.datasets.MNIST(
>     './data', train=False, download=True, transform=transform
> )
>
> train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
> test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
> ```
