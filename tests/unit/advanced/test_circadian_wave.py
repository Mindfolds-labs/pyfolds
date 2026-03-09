import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced


def test_circadian_outputs_present_when_enabled():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(3, 2, 4)

    out = neuron(x, collect_stats=False, dt=2.0)

    assert "circadian_phase" in out
    assert "circadian_embedding" in out
    assert out["circadian_embedding"].shape == (7,)
    assert out["circadian_meridiem"] in {"AM", "PM"}


def test_circadian_memory_store_and_recall():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    pattern = torch.randn(3)

    neuron.store_temporal_memory(pattern, importance=0.9, concept="luz")
    recalled = neuron.recall_temporal_memories(n=1)

    assert len(recalled) == 1
    assert torch.allclose(recalled[0].cpu(), pattern)
    assert neuron.recall_when("luz")


def test_circadian_temporal_age_and_consolidation_cycle():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(2, 2, 4)

    neuron(x, collect_stats=False, dt=3600.0)
    assert neuron.get_temporal_memory_stats()["age_seconds"] >= 3600

    neuron.store_temporal_memory(torch.randn(4), importance=0.95, concept="marco")
    neuron.store_temporal_memory(torch.randn(4), importance=0.05, concept="fraco")

    report = neuron.consolidate_temporal_memory(pruning_threshold=0.1)
    assert report["consolidated"] >= 1
    assert report["pruned"] >= 1

    story = neuron.narrative_of_life()
    assert "dias de vida" in story


def test_circadian_auto_mode_switches_online_and_sleep():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
        circadian_auto_mode=True,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    x = torch.randn(2, 2, 4)

    neuron.circadian_phase.fill_(10.0)
    out_am = neuron(x, collect_stats=False, dt=0.0)
    assert out_am["circadian_mode"] == "online"

    sleep_before = int(neuron.sleep_count.item())
    neuron.circadian_phase.fill_(190.0)
    out_pm = neuron(x, collect_stats=False, dt=0.0)
    assert out_pm["circadian_mode"] == "sleep"
    assert int(neuron.sleep_count.item()) == sleep_before + 1
