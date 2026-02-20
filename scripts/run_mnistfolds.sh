#!/usr/bin/env bash
set -euo pipefail
python train_mnist_folds.py --model py --epochs 10 --batch-size 128 --lr 1e-3 --run-id mnistfolds_py
python train_mnist_folds.py --model py --epochs 20 --batch-size 128 --lr 1e-3 --run-id mnistfolds_py --resume
