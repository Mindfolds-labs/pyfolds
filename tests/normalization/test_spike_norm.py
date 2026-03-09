import torch

from pyfolds.normalization.spike_norm import SpikeLayerNorm


def test_spike_norm_shape_and_numerics():
    norm = SpikeLayerNorm(8)
    x = torch.randn(4, 8)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
