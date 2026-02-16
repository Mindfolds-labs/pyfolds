from concurrent.futures import ThreadPoolExecutor

import pytest

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind


@pytest.mark.concurrency
def test_parallel_reads_same_fold_file(tmp_path):
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    neuron = MPJRDNeuron(cfg)
    file_path = tmp_path / "parallel.fold"

    save_fold_or_mind(
        neuron,
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=True,
        compress="none",
    )

    def read_once():
        with FoldReader(str(file_path), use_mmap=True) as reader:
            payload = reader.read_chunk_bytes("torch_state", verify=True)
            arrays = reader.read_chunk_bytes("nuclear_arrays", verify=True)
        return len(payload), len(arrays)

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda _: read_once(), range(10)))

    assert len(results) == 10
    assert len(set(results)) == 1
