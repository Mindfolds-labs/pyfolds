"""Tests for StatisticsAccumulator."""

import pytest
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
