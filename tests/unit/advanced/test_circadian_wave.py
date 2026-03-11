import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.utils.types import LearningMode


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


def test_circadian_plasticity_gate_varies_by_phase():
    """Gate de plasticidade deve ser maior em AM que em PM."""
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
        circadian_day_start_hour=0.0,
    )
    neuron = MPJRDNeuronAdvanced(cfg)

    ctx_am = {
        "phase": 0.0,
        "meridiem": "AM",
        "day": 0,
        "cortisol": 1.0,
        "melatonin": 0.1,
        "hour": 0.0,
        "focus_gain": 1.0,
        "recommended_mode": LearningMode.ONLINE,
    }
    gate_am = neuron._apply_circadian_plasticity_gate(ctx_am)

    ctx_pm = {
        "phase": 180.0,
        "meridiem": "PM",
        "day": 0,
        "cortisol": 0.3,
        "melatonin": 0.9,
        "hour": 6.0,
        "focus_gain": 0.3,
        "recommended_mode": LearningMode.SLEEP,
    }
    gate_pm = neuron._apply_circadian_plasticity_gate(ctx_pm)

    assert gate_am > gate_pm
    assert 0.0 < gate_pm < gate_am <= 2.0


def test_circadian_offline_consolidation_pipeline_tracks_audit():
    cfg = MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        wave_enabled=True,
        circadian_enabled=True,
        enable_sleep_consolidation=False,
    )
    neuron = MPJRDNeuronAdvanced(cfg)
    neuron.store_temporal_memory(torch.randn(4), importance=0.95, concept="forte")

    report = neuron.consolidate_memories(trigger="unit_test")
    assert report["executed"] is False

    stats = neuron.get_temporal_memory_stats()
    assert stats["offline_consolidation_requested"] == 1
    assert stats["offline_consolidation_skipped"] == 1
    assert stats["offline_consolidation_executed"] == 0
