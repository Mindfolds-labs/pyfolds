"""Diagnostic benchmark: dot-product attention vs resonance attention."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from pyfolds.leibreg import ResonanceAttention


def dot_attention(x: torch.Tensor) -> torch.Tensor:
    scale = x.shape[-1] ** -0.5
    logits = torch.matmul(x, x.transpose(-2, -1)) * scale
    attn = torch.softmax(logits, dim=-1)
    return torch.matmul(attn, x)


def run() -> None:
    device = torch.device("cpu")
    x = torch.randn(8, 64, 32, device=device, requires_grad=True)

    resonance = ResonanceAttention(dim=32).to(device)

    t0 = time.perf_counter()
    y_dot = dot_attention(x)
    loss_dot = F.mse_loss(y_dot, torch.zeros_like(y_dot))
    loss_dot.backward(retain_graph=True)
    t1 = time.perf_counter()
    dot_grad_ok = bool(torch.isfinite(x.grad).all().item())

    x.grad = None
    t2 = time.perf_counter()
    y_res = resonance(x)
    loss_res = F.mse_loss(y_res, torch.zeros_like(y_res))
    loss_res.backward()
    t3 = time.perf_counter()
    res_grad_ok = bool(torch.isfinite(x.grad).all().item())

    print("dot_runtime_s", round(t1 - t0, 6))
    print("resonance_runtime_s", round(t3 - t2, 6))
    print("dot_output_norm", round(float(y_dot.norm(dim=-1).mean().item()), 6))
    print("resonance_output_norm", round(float(y_res.norm(dim=-1).mean().item()), 6))
    print("dot_grad_finite", dot_grad_ok)
    print("resonance_grad_finite", res_grad_ok)


if __name__ == "__main__":
    run()
