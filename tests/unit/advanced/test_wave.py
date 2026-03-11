import torch

from pyfolds.advanced import WaveMixin
from pyfolds.core import MPJRDConfig, MPJRDNeuron
from pyfolds.utils.types import LearningMode


class WaveTestNeuron(WaveMixin, MPJRDNeuron):
    def __init__(self, cfg: MPJRDConfig):
        super().__init__(cfg)
        if cfg.wave_enabled:
            self._init_wave(cfg)

    def forward(self, x, **kwargs):
        mode = kwargs.get("mode", LearningMode.ONLINE)
        if hasattr(self, "oscillators"):
            self.set_wave_mode(mode)
            x_mod, wave_state = self._wave_modulate(x)
            out = super().forward(x_mod, **kwargs)
            self._wave_update(out["spikes"], out["u"])
            out.update(wave_state)
            out["wave_sync"] = self._compute_sync(out["u"])
            out.update(self.get_wave_metrics())
            return out
        return super().forward(x, **kwargs)


def test_wave_mixin_modulates_forward_and_metrics():
    cfg = MPJRDConfig(
        wave_enabled=True,
        wave_n_frequencies=4,
        n_dendrites=2,
        n_synapses_per_dendrite=4,
    )
    neuron = WaveTestNeuron(cfg)

    x = torch.ones(1, 2, 4)
    out = neuron(x, mode=LearningMode.ONLINE)

    assert "wave_modulation" in out
    assert "wave_sync" in out
    assert out["wave_n_frequencies"] == 4.0


def test_wave_mixin_sleep_consolidation_prunes_small_amplitudes():
    cfg = MPJRDConfig(wave_enabled=True, wave_n_frequencies=3, wave_sleep_pruning_threshold=0.05)
    neuron = WaveTestNeuron(cfg)

    with torch.no_grad():
        neuron.wave_amplitudes.copy_(torch.tensor([0.1, 0.001, 0.2]))

    neuron.set_wave_mode(LearningMode.SLEEP)

    assert neuron.wave_amplitudes[1].item() == 0.0


def test_wave_consolidation_pipeline_respects_toggle_and_tracks_audit():
    cfg = MPJRDConfig(
        wave_enabled=True,
        wave_n_frequencies=3,
        wave_sleep_pruning_threshold=0.05,
        enable_sleep_consolidation=False,
    )
    neuron = WaveTestNeuron(cfg)

    with torch.no_grad():
        neuron.wave_amplitudes.copy_(torch.tensor([0.1, 0.001, 0.2]))

    report = neuron.consolidate_memories(trigger="unit_test")
    assert report["executed"] is False
    assert neuron.wave_amplitudes[1].item() > 0.0

    metrics = neuron.get_wave_metrics()
    assert metrics["wave_consolidation_requested"] == 1.0
    assert metrics["wave_consolidation_skipped"] == 1.0
    assert metrics["wave_consolidation_executed"] == 0.0
