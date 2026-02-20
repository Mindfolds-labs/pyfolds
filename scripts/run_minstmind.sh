#!/usr/bin/env bash
set -euo pipefail
python train_mnist_mind.py --model wave --epochs 10 --batch-size 128 --lr 1e-3 --run-id minstmind_wave
python train_mnist_mind.py --model wave --epochs 20 --batch-size 128 --lr 1e-3 --run-id minstmind_wave --resume
