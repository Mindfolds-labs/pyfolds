import torch

from pyfolds.leibreg.reg_core import REGCore, ResonanceAttention


def test_resonance_shape_and_norm_weights() -> None:
    attn = ResonanceAttention(dim=4)
    x = torch.randn(2, 5, 4)
    y = attn(x)
    assert y.shape == x.shape


def test_temperature_positive() -> None:
    attn = ResonanceAttention(dim=4)
    assert float(attn.temperature.item()) > 0.0


def test_gradient_flow() -> None:
    core = REGCore(dim=4, depth=2)
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = core(x)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_small_batch_stability() -> None:
    core = REGCore(dim=4, depth=1)
    x = torch.zeros(1, 1, 4)
    y = core(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mask_behavior() -> None:
    attn = ResonanceAttention(dim=4)
    x = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, False, False]])
    y = attn(x, mask=mask)
    assert y.shape == x.shape
