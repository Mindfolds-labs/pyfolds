from pathlib import Path
import io

import numpy as np
import pyfolds

from pyfolds.serialization import FoldReader, load_fold_or_mind, peek_fold_or_mind, save_fold_or_mind


def test_fold_roundtrip_and_peek(tmp_path: Path):
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    neuron = pyfolds.MPJRDNeuron(cfg)

    path = tmp_path / "model.fold"
    save_fold_or_mind(neuron, str(path), tags={"run": "unit"})

    info = peek_fold_or_mind(str(path))
    assert "manifest" in info
    assert info["manifest"]["kind"] == "fold"
    assert "torch_state" in info["chunks"]
    assert "nuclear_arrays" in info["chunks"]

    restored = load_fold_or_mind(str(path), pyfolds.MPJRDNeuron)
    assert restored.cfg.n_dendrites == neuron.cfg.n_dendrites
    assert restored.cfg.n_synapses_per_dendrite == neuron.cfg.n_synapses_per_dendrite


def test_fold_reader_can_read_nuclear_arrays(tmp_path: Path):
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3)
    neuron = pyfolds.MPJRDNeuron(cfg)

    path = tmp_path / "model.mind"
    save_fold_or_mind(neuron, str(path), kind="mind")

    with FoldReader(str(path), use_mmap=True) as reader:
        raw = reader.read_chunk("nuclear_arrays")

    npz = np.load(io.BytesIO(raw))
    assert npz["N"].shape == (2, 3)
    assert npz["I"].shape == (2, 3)
