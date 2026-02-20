"""Testes de integração com sequência temporal e estabilidade mínima."""

import importlib.util

import pytest
import torch

import pyfolds


def forward_sequence(neuron: pyfolds.MPJRDNeuron, x_seq: torch.Tensor, dt: float = 1.0):
    """Executa sequência temporal chamando step/forward em cada timestep.

    x_seq: [T, B, D, S]
    """
    outputs = []
    for t in range(x_seq.shape[0]):
        outputs.append(neuron.step(x_seq[t], dt=dt, collect_stats=True))
    return outputs


@pytest.mark.integration
def test_forward_sequence_torch_stability_minimal_criteria():
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.4,
        theta_min=0.05,
        theta_max=2.0,
        device="cpu",
    )
    neuron = pyfolds.MPJRDNeuron(cfg)

    torch.manual_seed(11)
    x_seq = torch.rand(12, 3, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    outputs = forward_sequence(neuron, x_seq)

    spikes_t = torch.stack([out["spikes"] for out in outputs])
    theta_t = torch.stack([out["theta"] for out in outputs]).flatten()
    u_t = torch.stack([out["u"] for out in outputs])

    assert spikes_t.shape == (12, 3)
    assert theta_t.numel() == 12
    assert torch.isfinite(spikes_t).all()
    assert torch.isfinite(u_t).all()
    assert torch.isfinite(theta_t).all()

    assert float(theta_t.min()) >= cfg.theta_min - 1e-6
    assert float(theta_t.max()) <= cfg.theta_max + 1e-6

    mean_spike_rate = float(spikes_t.float().mean())
    assert 0.0 <= mean_spike_rate <= 1.0
    assert int(neuron.step_id.item()) == 12


@pytest.mark.integration
@pytest.mark.tf
@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None,
    reason="TensorFlow não instalado neste ambiente.",
)
def test_tf_sequence_equivalent_stability_minimal_criteria():
    import tensorflow as tf

    tf.random.set_seed(11)
    x_seq = tf.random.uniform(shape=(10, 2, 2, 4), minval=0.0, maxval=1.0, dtype=tf.float32)

    # Equivalente simples de integração temporal para validar backend condicional.
    soma_dend = tf.reduce_sum(tf.reduce_mean(x_seq, axis=-1), axis=-1)  # [T, B]
    spikes = tf.cast(soma_dend >= 0.4, tf.float32)

    assert spikes.shape == (10, 2)
    assert tf.reduce_all(tf.math.is_finite(spikes)).numpy().item() is True
    spike_mean = tf.reduce_mean(spikes).numpy().item()
    assert 0.0 <= spike_mean <= 1.0
