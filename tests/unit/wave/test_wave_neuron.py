import torch

from pyfolds.wave import MPJRDWaveConfig, MPJRDWaveNeuron


def test_wave_outputs_quadrature_and_phase_range():
    cfg = MPJRDWaveConfig(n_dendrites=2, n_synapses_per_dendrite=4, theta_init=0.1)
    neuron = MPJRDWaveNeuron(cfg)

    x = torch.ones(1, 2, 4)
    out = neuron(x, reward=0.8, target_class=3)

    assert "wave_real" in out and "wave_imag" in out
    assert out["phase"].min() >= 0
    assert out["phase"].max() <= torch.pi


def test_cooperative_integration_uses_multiple_dendrites():
    cfg = MPJRDWaveConfig(n_dendrites=2, n_synapses_per_dendrite=2, theta_init=1.2)
    neuron = MPJRDWaveNeuron(cfg)

    x = torch.zeros(1, 2, 2)
    x[:, 0, :] = 1.0
    x[:, 1, :] = 1.0

    out = neuron(x, reward=0.0)
    # soma cooperativa de ativações em dois dendritos (sem WTA)
    assert out["dendritic_activations"].shape == (1, 2)
    assert out["u"].item() > out["dendritic_activations"][0, 0].item()
