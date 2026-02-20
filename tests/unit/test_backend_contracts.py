"""Contratos de compatibilidade v1/v2 e backends (torch/tf)."""

import importlib.util

import numpy as np
import pytest
import torch

import pyfolds


def test_import_and_object_construction_v1_v2():
    """Garante compatibilidade de import/construção para v1 e v2."""
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3, device="cpu")

    neuron_v1 = pyfolds.MPJRDNeuron(cfg)
    neuron_v2 = pyfolds.MPJRDNeuronV2(cfg)

    assert isinstance(neuron_v1, pyfolds.MPJRDNeuron)
    assert isinstance(neuron_v2, pyfolds.MPJRDNeuronV2)


def test_torch_backend_shape_and_state_contracts():
    """Contrato mínimo de shape/estado no backend torch."""
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3, device="cpu")
    neuron_v1 = pyfolds.MPJRDNeuron(cfg)
    neuron_v2 = pyfolds.MPJRDNeuronV2(cfg)

    x = torch.ones(4, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    out_v1 = neuron_v1.forward(x, collect_stats=True)
    out_v2 = neuron_v2.forward(x, collect_stats=True)

    assert out_v1["spikes"].shape == (4,)
    assert out_v1["u"].shape == (4,)
    assert out_v1["v_dend"].shape == (4, cfg.n_dendrites)
    assert out_v1["theta"].shape == (1,)
    assert out_v1["r_hat"].shape == (1,)

    assert out_v2["spikes"].shape == (4,)
    assert out_v2["u"].shape == (4,)
    assert out_v2["somatic"].shape == (4,)
    assert out_v2["v_dend"].shape == (4, cfg.n_dendrites)
    assert out_v2["dendritic_gain"].shape == (4, cfg.n_dendrites)

    assert int(neuron_v1.step_id.item()) == 1
    assert int(neuron_v2.step_id.item()) == 1
    assert neuron_v1.N.shape == (cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    assert neuron_v1.I.shape == (cfg.n_dendrites, cfg.n_synapses_per_dendrite)




def test_torch_backend_v2_accepts_multidim_batch_contract():
    """V2 deve aceitar entradas com batch multidimensional [..., D, S]."""
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=3, device="cpu")
    neuron_v2 = pyfolds.MPJRDNeuronV2(cfg)

    x = torch.ones(2, 5, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    out_v2 = neuron_v2.forward(x, collect_stats=True)

    assert out_v2["spikes"].shape == (2, 5)
    assert out_v2["u"].shape == (2, 5)
    assert out_v2["somatic"].shape == (2, 5)
    assert out_v2["v_dend"].shape == (2, 5, cfg.n_dendrites)
    assert out_v2["dendritic_gain"].shape == (2, 5, cfg.n_dendrites)

def _tf_forward_sequence_equivalent(x_seq: "object") -> "object":
    """Equivalente simples de forward_sequence para validar shape/estabilidade em tf.

    x_seq: [T, B, D, S]
    retorna: [T, B]
    """
    import tensorflow as tf

    dendritic = tf.reduce_mean(x_seq, axis=-1)  # [T, B, D]
    somatic = tf.reduce_sum(dendritic, axis=-1)  # [T, B]
    return tf.cast(somatic >= 0.5, tf.float32)


@pytest.mark.tf
@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow não instalado neste ambiente.",
)
def test_tf_backend_conditional_shape_state_contracts():
    """Teste condicional de backend tf (shape, faixa e estabilidade numérica)."""
    import tensorflow as tf

    tf.random.set_seed(7)
    x_seq = tf.random.uniform(shape=(5, 3, 2, 4), minval=0.0, maxval=1.0, dtype=tf.float32)

    spikes = _tf_forward_sequence_equivalent(x_seq)

    assert spikes.shape == (5, 3)
    assert tf.reduce_all(tf.math.is_finite(spikes)).numpy().item() is True
    assert tf.reduce_all((spikes >= 0.0) & (spikes <= 1.0)).numpy().item() is True

    as_np = spikes.numpy()
    assert np.array_equal(as_np, as_np.astype(np.float32))
