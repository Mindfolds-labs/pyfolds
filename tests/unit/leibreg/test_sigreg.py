import torch

from pyfolds.leibreg.sigreg import SIGReg


def test_sigreg_scalar_and_grad() -> None:
    reg = SIGReg(dim=4, num_projections=16)
    q = torch.randn(6, 4, requires_grad=True)
    loss = reg(q)
    assert loss.ndim == 0
    loss.backward()
    assert q.grad is not None


def test_sigreg_numerical_stability_small_inputs() -> None:
    reg = SIGReg(dim=4, num_projections=8)
    q = torch.zeros(2, 4)
    loss = reg(q)
    assert torch.isfinite(loss)
