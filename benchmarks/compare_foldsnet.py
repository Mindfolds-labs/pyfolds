"""Benchmark simples entre FOLDSNet e baseline linear."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from foldsnet.factory import create_foldsnet


class LinearBaseline(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], n_classes: int):
        super().__init__()
        c, h, w = in_shape
        self.fc = nn.Linear(c * h * w, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.shape[0], -1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    folds = create_foldsnet("4L", args.dataset)
    if args.dataset == "mnist":
        in_shape, n_classes = (1, 28, 28), 10
    else:
        in_shape, n_classes = (3, 32, 32), 10
    baseline = LinearBaseline(in_shape, n_classes)

    x = torch.randn(8, *in_shape)
    y = torch.randint(0, n_classes, (8,))
    loss_fn = nn.CrossEntropyLoss()

    folds_loss = loss_fn(folds(x), y).item()
    base_loss = loss_fn(baseline(x), y).item()
    print(f"FOLDSNet loss: {folds_loss:.4f}")
    print(f"Linear baseline loss: {base_loss:.4f}")


if __name__ == "__main__":
    main()
