"""Tests for device utilities."""

import torch
import pyfolds
from pyfolds.utils import infer_device, ensure_device, get_device


class TestDevice:
    """Test device functions."""
    
    def test_infer_device(self):
        """Test device inference."""
        x = torch.randn(10)
        device = infer_device(x)
        assert device.type == 'cpu'
    
    def test_get_device(self):
        """Test get device."""
        device = get_device('cpu')
        assert device.type == 'cpu'
