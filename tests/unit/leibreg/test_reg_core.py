import torch

from pyfolds.leibreg.reg_core import REGCore


def test_reg_core_preserves_shape() -> None:
    model = REGCore(metric="euclidean", num_steps=2)
    x = torch.randn(2, 6, 4)
    out = model(x)
    assert out.shape == x.shape


def test_reg_core_supports_metrics() -> None:
    x = torch.randn(1, 4, 4)
    out_e = REGCore(metric="euclidean")(x)
    out_c = REGCore(metric="cosine")(x)
    assert out_e.shape == out_c.shape == x.shape


def test_reg_core_residual_stable() -> None:
    x = torch.randn(2, 5, 4)
    out = REGCore(residual=0.95, num_steps=4)(x)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 1e4


def test_reg_core_output_is_finite() -> None:
    x = torch.randn(3, 2, 4)
    out = REGCore(temperature=0.5)(x)
    assert torch.isfinite(out).all()
