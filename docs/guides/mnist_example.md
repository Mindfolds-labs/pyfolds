# Guia Prático — MNIST com PyFolds (MPJRD)

Este tutorial mostra um pipeline mínimo para treinar uma rede com neurônios MPJRD usando MNIST.

## 1. Pré-requisitos

```bash
pip install torch torchvision pyfolds
```

## 2. Objetivo

- Carregar MNIST.
- Transformar imagens para o formato esperado pelo MPJRD (`[B, D, S]` por neurônio/camada).
- Treinar uma rede pequena.
- Inspecionar métricas interpretáveis (`theta`, `N`, taxa de spike).

## 3. Script completo

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyfolds import MPJRDConfig, MPJRDLayer

# ---------- Configuração ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 3
lr = 1e-3

# Para MNIST (28x28=784), vamos mapear para [D=7, S=16] e usar 10 neurônios de saída
D, S = 7, 16
features = D * S  # 112
n_neurons = 10

cfg = MPJRDConfig(
    n_dendrites=D,
    n_synapses_per_dendrite=S,
    plastic=True,
    defer_updates=True,
    target_spike_rate=0.1,
)

layer = MPJRDLayer(n_neurons=n_neurons, cfg=cfg, name="mnist_out").to(device)
head = nn.Linear(n_neurons, 10).to(device)
optimizer = torch.optim.Adam(head.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ---------- Dados ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)


def preprocess(images: torch.Tensor) -> torch.Tensor:
    """
    images: [B, 1, 28, 28] -> [B, n_neurons, D, S]
    Estratégia simples: downsample para 14x8=112 e replica para cada neurônio.
    """
    x = torch.nn.functional.interpolate(images, size=(14, 8), mode="bilinear", align_corners=False)
    x = x.flatten(1)  # [B, 112]
    x = x.view(-1, D, S)  # [B, D, S]
    x = x.unsqueeze(1).repeat(1, n_neurons, 1, 1)  # [B, 10, D, S]
    return x


# ---------- Treino ----------
for epoch in range(1, epochs + 1):
    layer.train()
    head.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        x = preprocess(images)
        out = layer(x, reward=0.1)

        spikes = out["spikes"]  # [B, 10]
        logits = head(spikes)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Consolida atualização local no MPJRD
        for neuron in layer.neurons:
            neuron.apply_plasticity(dt=1.0, reward=0.1)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)

    train_loss = total_loss / total
    train_acc = total_correct / total

    # ---------- Validação ----------
    layer.eval()
    head.eval()
    correct = 0
    count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            x = preprocess(images)

            out = layer(x, reward=None)
            logits = head(out["spikes"])
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += images.size(0)

    test_acc = correct / count

    # Métricas interpretáveis
    theta_mean = out["thetas"].mean().item()
    rate_mean = out["rates"].mean().item()

    print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f} | theta={theta_mean:.3f} | rate={rate_mean:.3f}")

# Fase de sono opcional
for neuron in layer.neurons:
    neuron.sleep(duration=30.0)
```

## 4. O que esperar dos resultados

- Acurácia inicial baixa, subindo gradualmente por época.
- `theta` tende a se ajustar para manter `target_spike_rate`.
- Com `defer_updates=True`, o modelo tende a ficar mais estável entre batches.

## 5. Próximos passos

- Substituir `head` linear por múltiplas camadas.
- Ajustar `n_dendrites` e `n_synapses_per_dendrite` para maior capacidade.
- Testar `neuromod_mode="capacity"` para regular saturação.
