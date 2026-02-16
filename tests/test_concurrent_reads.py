import pytest
from concurrent.futures import ThreadPoolExecutor

from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind


def _build_neuron() -> MPJRDNeuron:
from pyfolds.core.config import MPJRDConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind


@pytest.mark.concurrency
def test_parallel_reads_same_fold_file(tmp_path):
 main
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg)


def _read_fold_snapshot(path: str):
    with FoldReader(path, use_mmap=True) as reader:
        return {
            "chunks": tuple(reader.list_chunks()),
            "torch_state": reader.read_chunk_bytes("torch_state", verify=True),
            "manifest": reader.read_chunk_bytes("llm_manifest", verify=True),
        }


@pytest.mark.concurrency
def test_concurrent_reads_same_fold_file_consistent(tmp_path):
    file_path = tmp_path / "parallel.fold"
    save_fold_or_mind(
        _build_neuron(),
        str(file_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=False,
        compress="none",
    )

    baseline = _read_fold_snapshot(str(file_path))

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: _read_fold_snapshot(str(file_path)), range(10)))

    for result in results:
        assert result["chunks"] == baseline["chunks"]
        assert result["torch_state"] == baseline["torch_state"]
        assert result["manifest"] == baseline["manifest"]

        main
