import torch
import pyfolds


def test_forward_clamps_non_finite_dendritic_sum(monkeypatch):
    cfg = pyfolds.NeuronConfig()
    neuron = pyfolds.MPJRDNeuron(cfg)

    def bad_forward(self, x):
        return torch.full((x.shape[0],), float('nan'), device=x.device)

    monkeypatch.setattr(
        neuron.dendrites[0],
        "forward",
        bad_forward.__get__(neuron.dendrites[0], type(neuron.dendrites[0])),
    )

    x = torch.ones(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)
    out = neuron.forward(x, collect_stats=False)

    assert torch.isfinite(out["v_dend"]).all()
