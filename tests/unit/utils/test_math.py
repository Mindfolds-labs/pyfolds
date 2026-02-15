"""Tests for math utilities."""

import torch
import pyfolds
from pyfolds.utils import safe_div, clamp_rate, clamp_R


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
