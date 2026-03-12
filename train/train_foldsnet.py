"""Script de treinamento principal da FOLDSNet."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from foldsnet.factory import create_foldsnet


def _load_dataset(name: str, batch_size: int) -> DataLoader:
    tfm = transforms.ToTensor()
    if name == "mnist":
        ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    elif name == "cifar10":
        ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
    else:
        raise ValueError("Dataset inválido. Use mnist ou cifar10.")
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="4L")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_format", choices=["fold", "mind"], default="fold")
    args = parser.parse_args()

    model = create_foldsnet(args.variant, args.dataset)
    train_loader = _load_dataset(args.dataset, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.save(f"checkpoints/epoch_{epoch}.{args.save_format}", format=args.save_format)

    model.save(f"models/final_prod.{args.save_format}", format=args.save_format)


if __name__ == "__main__":
    main()
