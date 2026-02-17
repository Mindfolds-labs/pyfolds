"""Tests for Neuromodulator."""

import pytest
import torch
import pyfolds


class TestNeuromodulator:
    """Test neuromodulator."""
    
    def test_external_mode(self):
        """Test external mode."""
        cfg = pyfolds.MPJRDConfig(neuromod_mode="external")
        from pyfolds.core import Neuromodulator
        
        mod = Neuromodulator(cfg)
        R = mod(rate=0.1, r_hat=0.1, reward=0.5)
        
        assert R.item() == 0.5
    
    def test_surprise_mode(self):
        """Test surprise mode."""
        cfg = pyfolds.MPJRDConfig(
            neuromod_mode="surprise",
            sup_k=2.0
        )
        from pyfolds.core import Neuromodulator
        
        mod = Neuromodulator(cfg)
        R = mod(rate=0.2, r_hat=0.1)
        
        assert abs(R.item() - 0.2) < 1e-6


    def test_rejects_non_finite_inputs(self):
        """Non-finite scalars must fail fast to avoid unstable dynamics."""
        cfg = pyfolds.MPJRDConfig(neuromod_mode="surprise")
        from pyfolds.core import Neuromodulator

        mod = Neuromodulator(cfg)

        with pytest.raises(ValueError, match="rate deve ser finito"):
            mod(rate=float("nan"), r_hat=0.1)

        with pytest.raises(ValueError, match="r_hat deve ser finito"):
            mod(rate=0.1, r_hat=float("inf"))

    def test_infers_output_device_from_tensor_inputs(self):
        """Output tensor should stay on same device when device arg is omitted."""
        cfg = pyfolds.MPJRDConfig(neuromod_mode="surprise")
        from pyfolds.core import Neuromodulator

        mod = Neuromodulator(cfg)
        inp = torch.tensor([0.2])
        out = mod(rate=inp, r_hat=inp)

        assert out.device == inp.device
