"""Tests for StatisticsAccumulator."""

from collections import deque

import pytest
import torch


class TestStatisticsAccumulator:
    def test_dense_baseline_updates_and_averages(self):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3, mode="dense")
        x = torch.tensor(
            [
                [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]],
                [[3.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            ]
        )
        gated = torch.ones(2, 2)
        spikes = torch.tensor([1.0, 0.0])

        acc.accumulate(x, gated, spikes)
        stats = acc.get_averages()

        assert stats.total_samples == 2
        assert torch.allclose(stats.x_mean, x.mean(dim=0))
        assert torch.allclose(acc.synapse_sample_count, torch.full((2, 3), 2.0))
        assert stats.post_rate == pytest.approx(0.5)

    def test_sparse_masked_only_updates_active_positions(self):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(
            1,
            4,
            mode="sparse_masked",
            activity_threshold=0.5,
            sparse_min_activity_ratio=1.0,
        )
        x = torch.tensor([[[0.8, 0.1, -0.9, 0.01]]])
        gated = torch.tensor([[1.0]])
        spikes = torch.tensor([1.0])

        acc.accumulate(x, gated, spikes)

        assert torch.allclose(acc.acc_x, torch.tensor([[0.8, 0.0, -0.9, 0.0]]))
        assert torch.allclose(acc.synapse_sample_count, torch.tensor([[1.0, 0.0, 1.0, 0.0]]))

    def test_sparse_fallback_to_dense_when_activity_high(self):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(
            1,
            4,
            mode="sparse_masked",
            activity_threshold=0.05,
            sparse_min_activity_ratio=0.25,
        )
        x = torch.ones(2, 1, 4)
        gated = torch.ones(2, 1)
        spikes = torch.tensor([0.0, 1.0])

        acc.accumulate(x, gated, spikes)

        assert acc.dense_fallback_used is True
        assert acc.sparse_path_used is False
        assert torch.allclose(acc.synapse_sample_count, torch.full((1, 4), 2.0))

    def test_gated_path_and_per_synapse_count_with_sparse_mode(self):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(
            2,
            2,
            mode="sparse_masked",
            activity_threshold=0.1,
            sparse_min_activity_ratio=1.0,
        )
        x = torch.tensor([[[0.2, 0.0], [0.0, 0.3]]])
        gated = torch.tensor([[0.7, 0.8]])
        spikes = torch.tensor([1.0])

        acc.accumulate(x, gated, spikes)
        stats = acc.get_averages()

        assert torch.allclose(acc.synapse_sample_count, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        assert torch.allclose(stats.gated_mean, torch.tensor([0.7, 0.8]))

    def test_sparse_and_dense_equivalence_with_full_activity(self):
        from pyfolds.core import StatisticsAccumulator

        x = torch.rand(8, 2, 3)
        gated = torch.rand(8, 2)
        spikes = torch.randint(0, 2, (8,), dtype=torch.float32)

        dense = StatisticsAccumulator(2, 3, mode="dense")
        sparse = StatisticsAccumulator(
            2,
            3,
            mode="sparse_masked",
            activity_threshold=0.0,
            sparse_min_activity_ratio=0.0,
        )

        dense.accumulate(x, gated, spikes)
        sparse.accumulate(x, gated, spikes)

        d_stats = dense.get_averages()
        s_stats = sparse.get_averages()
        assert torch.allclose(d_stats.x_mean, s_stats.x_mean, atol=1e-6, rtol=1e-6)
        assert torch.allclose(d_stats.gated_mean, s_stats.gated_mean, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_edge_cases_zero_near_threshold_large_values(self, dtype):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(
            1,
            3,
            mode="sparse_masked",
            activity_threshold=0.01,
            sparse_min_activity_ratio=1.0,
        )
        x = torch.tensor([[[0.0, 0.0100001, 1e6]]], dtype=dtype)
        gated = torch.tensor([[1.0]], dtype=dtype)
        spikes = torch.tensor([0.0], dtype=dtype)

        acc.accumulate(x, gated, spikes)
        stats = acc.get_averages()

        assert torch.isfinite(stats.x_mean).all()
        assert torch.isfinite(stats.gated_mean).all()

    def test_track_extra_accumulates_scalars_and_vectors(self):
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

    def test_history_respects_max_len(self):
        from pyfolds.core import StatisticsAccumulator

        acc = StatisticsAccumulator(2, 3, max_history_len=5)
        acc.enable_history(True)

        x = torch.ones(2, 2, 3)
        gated = torch.ones(2, 2)
        spikes = torch.tensor([1.0, 0.0])

        for _ in range(12):
            acc.accumulate(x, gated, spikes)

        assert len(acc.history["spike_rate"]) == 5
        assert len(acc.history["sparsity"]) == 5
        assert isinstance(acc.history["spike_rate"], deque)
