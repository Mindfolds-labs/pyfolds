"""Tests for StatisticsAccumulator."""

import pytest
from collections import deque
import torch
import pyfolds


class TestStatisticsAccumulator:
    """Test accumulator."""
    
    def test_initialization(self):
        """Test initialization."""
        from pyfolds.core import StatisticsAccumulator
        
        acc = StatisticsAccumulator(2, 3)
        assert acc.acc_x.shape == (2, 3)
        assert acc.acc_gated.shape == (2,)
    
    def test_accumulate(self):
        """Test accumulation."""
        from pyfolds.core import StatisticsAccumulator
        
        acc = StatisticsAccumulator(2, 3)
        x = torch.ones(4, 2, 3)
        gated = torch.ones(4, 2)
        spikes = torch.tensor([1., 0., 1., 0.])
        
        acc.accumulate(x, gated, spikes)
        
        stats = acc.get_averages()
        assert stats.total_samples == 4
        assert stats.post_rate == 0.5


    def test_history_respects_max_len(self):
        """History must not grow unbounded when enabled."""
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3, max_history_len=5)
        acc.enable_history(True)

        x = torch.ones(2, 2, 3)
        gated = torch.ones(2, 2)
        spikes = torch.tensor([1., 0.])

        for _ in range(12):
            acc.accumulate(x, gated, spikes)

        assert len(acc.history['spike_rate']) == 5
        assert len(acc.history['sparsity']) == 5
        assert isinstance(acc.history['spike_rate'], deque)

    def test_accumulate_rejects_invalid_shape(self):
        """Accumulator should fail fast when input shape is invalid."""
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3)
        x = torch.ones(4, 2, 4)  # last dim mismatch
        gated = torch.ones(4, 2)
        spikes = torch.zeros(4)

        with pytest.raises(ValueError, match=r"Esperado \[B, 2, 3\]"):
            acc.accumulate(x, gated, spikes)

    def test_track_extra_accumulates_scalars_and_vectors(self):
        """Extra tracked signals should be averaged correctly across batches."""
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3, track_extra=True)

        x = torch.ones(2, 2, 3)
        gated = torch.tensor([[1.0, 0.0], [0.5, 0.0]])
        spikes = torch.tensor([1.0, 0.0])
        v_dend = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = torch.tensor([0.5, 1.5])
        theta = torch.tensor(2.0)
        r_hat = torch.tensor([0.2, 0.4])
        adaptation = torch.tensor(1.0)

        acc.accumulate(x, gated, spikes, v_dend=v_dend, u=u, theta=theta, r_hat=r_hat, adaptation=adaptation)
        acc.accumulate(x, gated, spikes, v_dend=v_dend, u=u, theta=theta, r_hat=r_hat, adaptation=adaptation)

        stats = acc.get_averages()
        assert stats.total_samples == 4
        assert stats.theta_mean == pytest.approx(2.0)
        assert stats.u_mean == pytest.approx(1.0)
        assert stats.r_hat_mean == pytest.approx(0.3)
        assert stats.adaptation_mean == pytest.approx(1.0)
        assert torch.allclose(stats.v_dend_mean, torch.tensor([2.0, 3.0]))

    def test_reset_clears_state_and_history(self):
        """Reset should clear accumulated values and optional history."""
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3)
        acc.enable_history(True)

        x = torch.ones(2, 2, 3)
        gated = torch.ones(2, 2)
        spikes = torch.tensor([1.0, 0.0])
        acc.accumulate(x, gated, spikes)

        assert acc.has_data
        assert len(acc.history["spike_rate"]) == 1

        acc.reset()

        assert not acc.has_data
        assert acc.batch_count == 0
        assert len(acc.history["spike_rate"]) == 0
