# tests/performance/test_memory_usage.py
import psutil
import os

class TestMemoryUsage:
    def test_memory_leak(self, small_config):
        """Testa se há vazamento de memória."""
        import tracemalloc
        tracemalloc.start()
        
        neuron = pyfolds.MPJRDNeuron(small_config)
        # ... executa muitas iterações
        
        current, peak = tracemalloc.get_traced_memory()
        assert peak < 500 * 1024 * 1024  # < 500MB