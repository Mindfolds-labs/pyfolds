"""Smoke tests de corrupção para o plano operacional A3.

Cobertura aprofundada vive em `tests/unit/serialization/test_foldio.py`.
"""

import pytest

from pyfolds import NeuronConfig
from pyfolds.core.neuron import MPJRDNeuron
from pyfolds.serialization import FoldReader, save_fold_or_mind


def test_fold_corruption_is_detected(tmp_path):
    cfg = NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4, device="cpu")
    neuron = MPJRDNeuron(cfg)
    target = tmp_path / "corrupted.fold"

    save_fold_or_mind(neuron, str(target), compress="none")

    with FoldReader(str(target), use_mmap=False) as reader:
        chunk = next(c for c in reader.index["chunks"] if c["name"] == "torch_state")
        pos = chunk["offset"] + chunk["header_len"] + 1

    with open(target, "r+b") as fh:
        fh.seek(pos)
        original = fh.read(1)
        fh.seek(pos)
        fh.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(RuntimeError):
        with FoldReader(str(target), use_mmap=False) as reader:
            reader.read_chunk_bytes("torch_state", verify=True)
