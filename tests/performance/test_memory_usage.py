"""Performance tests for memory footprint."""

import tracemalloc

import torch
import pyfolds


class TestMemoryUsage:
    def test_memory_leak(self, small_config):
        """Verifica limite de uso de memória de pico em workload curto."""
        tracemalloc.start()

        neuron = pyfolds.MPJRDNeuron(small_config)

        # Workload curto e determinístico para detectar crescimento anômalo.
        x = torch.full((4, small_config.n_dendrites, small_config.n_synapses_per_dendrite), 0.5)
        for _ in range(100):
            neuron.forward(x)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 500 * 1024 * 1024  # < 500MB
