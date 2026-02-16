from concurrent.futures import ThreadPoolExecutor

import pytest

<<<<<<< codex/add-test-cases-for-corruption-detection
from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind
=======
from pyfolds.serialization import FoldReader, save_fold_or_mind
from tests.unit.serialization.test_foldio import _build_neuron
>>>>>>> main


@pytest.mark.concurrency
def test_parallel_reads_same_fold_file(tmp_path):
<<<<<<< codex/add-test-cases-for-corruption-detection
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
=======
    neuron = _build_neuron()
    file_path = tmp_path / "shared.fold"

    save_fold_or_mind(neuron, str(file_path), compress="none", include_history=False, include_telemetry=False)

    def read_once() -> tuple[set[str], int]:
        with FoldReader(str(file_path), use_mmap=True) as reader:
            chunks = set(reader.list_chunks())
            payload = reader.read_chunk_bytes("torch_state", verify=True)
            return chunks, len(payload)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: read_once(), range(10)))

    for chunks, payload_len in results:
        assert "torch_state" in chunks
        assert "llm_manifest" in chunks
        assert payload_len > 0
>>>>>>> main
