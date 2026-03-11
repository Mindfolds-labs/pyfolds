import torch
import pyfolds


def test_time_counter_increments_at_end_of_forward_step():
    cfg = pyfolds.NeuronConfig(t_refrac_abs=2.0)
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    out0 = neuron.forward(x, dt=1.0)
    assert neuron.time_counter.item() == 1.0

    out1 = neuron.forward(x, dt=1.0)
    assert out1["refrac_blocked"][0].item() is True
    assert neuron.time_counter.item() == 2.0

    assert out0["spikes"][0].item() == 1.0
    assert out1["spikes"][0].item() == 0.0


def test_time_counter_single_increment():
    cfg = pyfolds.NeuronConfig(wave_enabled=True)
    x = torch.ones(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    advanced = pyfolds.MPJRDNeuronAdvanced(cfg)
    wave_advanced = pyfolds.MPJRDWaveNeuronAdvanced(cfg)

    t0_adv = advanced.time_counter.item()
    out_adv = advanced.forward(x, dt=1.0)
    t1_adv = advanced.time_counter.item()

    t0_wave = wave_advanced.time_counter.item()
    out_wave = wave_advanced.forward(x, dt=1.0)
    t1_wave = wave_advanced.time_counter.item()

    assert t1_adv - t0_adv == 1.0
    assert t1_wave - t0_wave == 1.0

    # Guarda de regressão: cadeia completa de mixins deve executar sem
    # provocar incremento temporal duplicado por chamada encadeada.
    assert "refrac_blocked" in out_adv
    assert "dendrite_amplification" in out_adv
    assert "refrac_blocked" in out_wave
    assert "dendrite_amplification" in out_wave


def test_time_counter_increment_once_guard_per_step():
    cfg = pyfolds.NeuronConfig()
    neuron = pyfolds.MPJRDNeuronAdvanced(cfg)

    neuron._begin_time_step()
    neuron._increment_time_once(dt=1.0)
    neuron._increment_time_once(dt=1.0)

    assert neuron.time_counter.item() == 1.0

    neuron._begin_time_step()
    neuron._increment_time_once(dt=2.0)
    assert neuron.time_counter.item() == 3.0
