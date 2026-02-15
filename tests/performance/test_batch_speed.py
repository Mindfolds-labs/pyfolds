"""Performance tests for batch processing."""

import pytest
import torch
import time
import pyfolds


@pytest.mark.performance
@pytest.mark.slow
class TestBatchSpeed:
    """Test batch processing speed."""
    
    def test_forward_speed(self, small_config):
        """Measure forward pass speed."""
        neuron = pyfolds.MPJRDNeuron(small_config)
        
        batch_sizes = [1, 4, 16, 64]
        times = []
        
        for bs in batch_sizes:
            x = torch.randn(bs, 2, 4)
            
            start = time.perf_counter()
            for _ in range(100):
                out = neuron(x)
            end = time.perf_counter()
            
            times.append((end - start) / 100)
        
        # Just verify it runs, no assertion
        assert len(times) == 4
