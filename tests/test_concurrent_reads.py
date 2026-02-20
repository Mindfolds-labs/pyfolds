import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from pyfolds import NeuronConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind


def _build_neuron() -> MPJRDNeuron:
    cfg = NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        device="cpu",
        defer_updates=True,
        plastic=True,
    )
    return MPJRDNeuron(cfg)


def _read_signature(path: str):
    with FoldReader(path, use_mmap=True) as reader:
        manifest = reader.read_json("llm_manifest")
        torch_payload = reader.read_chunk_bytes("torch_state", verify=True)
        return {
            "hash": hashlib.sha256(torch_payload).hexdigest(),
            "step_id": manifest["expression"]["step_id"],
            "resume": manifest["routing"]["resume_training"],
            "n_chunks": len(reader.index.get("chunks", [])),
        }


@pytest.mark.concurrency
def test_parallel_reads_are_consistent(tmp_path):
    fold_path = tmp_path / "concurrent.fold"
    neuron = _build_neuron()

    save_fold_or_mind(
        neuron,
        str(fold_path),
        include_history=False,
        include_telemetry=False,
        include_nuclear_arrays=True,
        compress="none",
    )

    expected = _read_signature(str(fold_path))

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_read_signature, str(fold_path)) for _ in range(10)]

    results = [f.result() for f in futures]

    assert len(results) == 10
    assert all(r == expected for r in results)
