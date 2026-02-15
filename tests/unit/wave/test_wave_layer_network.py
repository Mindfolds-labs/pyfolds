import torch

from pyfolds import MPJRDWaveConfig, MPJRDWaveLayer, MPJRDWaveNetwork


def test_wave_layer_exposes_wave_outputs():
    cfg = MPJRDWaveConfig(n_dendrites=2, n_synapses_per_dendrite=4, theta_init=0.1)
    layer = MPJRDWaveLayer(n_neurons=3, cfg=cfg)

    x = torch.ones(2, 3, 2, 4)
    out = layer(x, neuron_kwargs={"target_class": 2})

    assert out["spikes"].shape == (2, 3)
    assert out["wave_real"].shape == (2, 3)
    assert out["wave_imag"].shape == (2, 3)
    assert out["phase"].shape == (2, 3)


def test_wave_network_forwards_layer_kwargs():
    cfg = MPJRDWaveConfig(n_dendrites=2, n_synapses_per_dendrite=4, theta_init=0.1)

    net = MPJRDWaveNetwork("wave_net")
    net.add_wave_layer("in", n_neurons=2, cfg=cfg)
    net.add_wave_layer("out", n_neurons=2, cfg=cfg)
    net.connect("in", "out")
    net.build()

    x = torch.ones(1, 2, 2, 4)
    out = net(
        x,
        layer_kwargs={
            "in": {"neuron_kwargs": {"target_class": 1}},
            "out": {"neuron_kwargs": {"target_class": 4}},
        },
    )

    assert out["output"].shape == (1, 2)
    assert "wave_real" in out["layers"]["out"]
