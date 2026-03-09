"""Lightweight diagnostic: dot-product attention vs proximity attention."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from pyfolds.leibreg import ProximityAttention


def dot_attention(x: torch.Tensor) -> torch.Tensor:
    scale = x.shape[-1] ** -0.5
    logits = torch.matmul(x, x.transpose(-2, -1)) * scale
    attn = torch.softmax(logits, dim=-1)
    return torch.matmul(attn, x)


def run() -> None:
    device = torch.device("cpu")
    x = torch.randn(8, 64, 32, device=device, requires_grad=True)

    prox = ProximityAttention(dim=32, kernel="gaussian", temperature=1.0).to(device)

    t0 = time.perf_counter()
    y_dot = dot_attention(x)
    loss_dot = F.mse_loss(y_dot, torch.zeros_like(y_dot))
    loss_dot.backward(retain_graph=True)
    t1 = time.perf_counter()

    x.grad = None
    t2 = time.perf_counter()
    y_prox = prox(x)
    loss_prox = F.mse_loss(y_prox, torch.zeros_like(y_prox))
    loss_prox.backward()
    t3 = time.perf_counter()

    print("dot_runtime_s", round(t1 - t0, 6))
    print("proximity_runtime_s", round(t3 - t2, 6))
    print("dot_output_std", round(float(y_dot.std().item()), 6))
    print("proximity_output_std", round(float(y_prox.std().item()), 6))
    print("dot_grad_finite", bool(torch.isfinite(x.grad).all().item()))


if __name__ == "__main__":
    run()
