# Treino MNIST (folds/mind)

Padrão de execução: **sem console por padrão**, com logs sempre em arquivo.

## Treino padrão (arquivo de log sempre ON)

### Backend folds
```bash
python train_mnist_folds.py --model py --epochs 10 --batch-size 128 --lr 1e-3 --run-id exp01
```

### Backend mind
```bash
python train_mnist_mind.py --model wave --epochs 10 --batch-size 128 --lr 1e-3 --run-id exp02
```

## Resume (checkpoint)

```bash
python train_mnist_folds.py --model py --epochs 20 --batch-size 128 --lr 1e-3 --run-id exp01 --resume
python train_mnist_mind.py --model wave --epochs 20 --batch-size 128 --lr 1e-3 --run-id exp02 --resume
```

## Parâmetros principais

- `--epochs N`
- `--batch-size N` (alias: `--batch`)
- `--lr X`
- `--run-id ID`
- `--resume`
- `--device {cpu,cuda}`
- `--console` (opcional)
- `--log-level INFO|DEBUG|...`
- `--log-file train.log` (relativo a `runs/<run-id>/`)
- `--sheer-cmd "<comando>"` (opcional; stdout/stderr vai para `train.log`)

## Logs e artefatos

Sempre em `runs/<run-id>/`:
- `train.log`
- `metrics.jsonl`
- `summary.json`
- `checkpoint.pt`
- `model.fold` ou `model.mind`

Em erro, cria automaticamente:
- `runs/<run-id>/ADR-ERR-<timestamp>.md`
- `docs/issues/BUG-<timestamp>.md`

## Scripts

```bash
bash scripts/run_mnistfolds.sh
bash scripts/run_minstmind.sh
```

## PowerShell

```powershell
.\scripts\run_mnist.ps1 -RunId mnist_global_01 -Backend folds -Model py -Epochs 10 -BatchSize 128 -Lr 0.001
.\scripts\run_mnist.ps1 -RunId mnist_global_01 -Backend folds -Model py -Epochs 20 -BatchSize 128 -Lr 0.001 -Resume
```
