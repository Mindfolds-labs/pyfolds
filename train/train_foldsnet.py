"""Treinamento básico da FOLDSNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from foldsnet.factory import create_foldsnet


def _fake_loader(shape: tuple[int, int, int], n_classes: int, batch_size: int) -> DataLoader:
    x = torch.randn(512, *shape)
    y = torch.randint(0, n_classes, (512,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Treino da FOLDSNet")
    parser.add_argument("--variant", default="4L")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_format", choices=["fold", "mind"], default="fold")
    args = parser.parse_args()

    model = create_foldsnet(args.variant, args.dataset)
    model.train()

    loader = _fake_loader(model.input_shape, model.n_classes, args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        for x, y in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    out_path = "models/final_prod.fold" if args.save_format == "fold" else "runs/final.mind"
    model.save(out_path, format=args.save_format, include_metadata=True)
    print(f"✅ Modelo salvo em {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
