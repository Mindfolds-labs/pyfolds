"""Tests for math utilities."""

import torch
import pyfolds
from pyfolds.utils import safe_div, clamp_rate, clamp_R
from pyfolds.utils.math import safe_weight_law


class TestMath:
    """Test math functions."""
    
    def test_safe_div(self):
        """Test safe division."""
        x = torch.tensor([10.0])
        y = torch.tensor([0.0])
        
        result = safe_div(x, y)
        assert result.item() > 1e8
    
    def test_clamp_rate(self):
        """Test rate clamping."""
        r = torch.tensor([-0.5, 0.3, 1.5])
        clamped = clamp_rate(r)
        
        expected = torch.tensor([0.0, 0.3, 1.0])
        assert torch.allclose(clamped, expected)
    
    def test_clamp_R(self):
        """Test neuromodulator clamping."""
        r = torch.tensor([-2.0, 0.5, 2.0])
        clamped = clamp_R(r)
        
        expected = torch.tensor([-1.0, 0.5, 1.0])
        assert torch.allclose(clamped, expected)

    def test_safe_weight_law_is_bounded_and_finite(self):
        """Safe weight law should cap very large N and avoid NaN/Inf."""
        n = torch.tensor([0, 31, 10**9], dtype=torch.int64)
        out = safe_weight_law(n, w_scale=5.0, max_log_val=4.0)

        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 4.0
