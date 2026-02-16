"""Benchmark inicial para roundtrip de serialização .fold.

Executar com:
    pytest benchmarks --benchmark-only --benchmark-json=benchmark.json
"""

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import load_fold_or_mind, save_fold_or_mind


def test_bench_foldio_roundtrip(tmp_path, benchmark):
    cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=8, device="cpu")
    neuron = MPJRDNeuron(cfg)
    path = tmp_path / "bench.fold"

    def _run():
        save_fold_or_mind(neuron, str(path), compress="none")
        _ = load_fold_or_mind(str(path), MPJRDNeuron, map_location="cpu")

    benchmark(_run)
