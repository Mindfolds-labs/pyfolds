# Comandos prontos (copiar e colar) — FOLDSNet

Este arquivo traz o fluxo completo para:
1. iniciar treino em MNIST,
2. continuar treino em CIFAR usando os pesos já treinados.

## 1) Treino inicial MNIST (gera checkpoint)

```bash
python train_mnist_folds.py \
  --model foldsnet \
  --foldsnet-variant 4L \
  --foldsnet-dataset mnist \
  --epochs 10 \
  --batch 64 \
  --lr 0.001 \
  --device cpu \
  --console
```

> O run-id será gerado automaticamente no formato `foldsnet_mnist_xxxxxx`.

## 2) Continuar treino no MESMO run (resume)

Use quando quiser continuar exatamente do último checkpoint do mesmo `run-id`.

```bash
python train_mnist_folds.py \
  --model foldsnet \
  --foldsnet-variant 4L \
  --foldsnet-dataset mnist \
  --epochs 20 \
  --batch 64 \
  --lr 0.001 \
  --device cpu \
  --run-id <RUN_ID_ANTERIOR> \
  --resume \
  --console
```

## 3) Transferência MNIST -> CIFAR10 (warm start por pesos)

Use quando quiser iniciar novo run em CIFAR10 carregando os pesos do MNIST.

```bash
python train_mnist_folds.py \
  --model foldsnet \
  --foldsnet-variant 4L \
  --foldsnet-dataset cifar10 \
  --epochs 10 \
  --batch 64 \
  --lr 0.0005 \
  --device cpu \
  --init-checkpoint runs/<RUN_ID_MNIST>/checkpoint.pt \
  --console
```

## 4) Transferência MNIST -> CIFAR100 (warm start por pesos)

```bash
python train_mnist_folds.py \
  --model foldsnet \
  --foldsnet-variant 4L \
  --foldsnet-dataset cifar100 \
  --epochs 10 \
  --batch 64 \
  --lr 0.0005 \
  --device cpu \
  --init-checkpoint runs/<RUN_ID_MNIST>/checkpoint.pt \
  --console
```

## Onde ficam os arquivos

Cada run gera pasta em `runs/<run_id>/` com:
- `checkpoint.pt` (pesos + otimizador para resume),
- `run_metadata.json` (configuração completa + timestamp),
- `metrics.jsonl`, `summary.json`, `train.log`,
- e opcionalmente `model.fold` / `model.mind`.

