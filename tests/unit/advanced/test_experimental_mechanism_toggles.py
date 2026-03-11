import copy
import io

import torch

from pyfolds import MPJRDConfig
from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds.advanced.experimental import (
    MechanismToggleSet,
    compare_mechanism_vs_baseline,
)


def _make_cfg(**extra) -> MPJRDConfig:
    return MPJRDConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        theta_init=0.1,
        plasticity_mode="stdp",
        spike_threshold=0.2,
        stdp_trace_threshold=0.0,
        **extra,
    )


def _make_input() -> torch.Tensor:
    torch.manual_seed(13)
    return torch.rand(3, 2, 4)


def _stdp_eligibility_tensor(neuron: MPJRDNeuronAdvanced) -> torch.Tensor:
    rows = []
    for dend in neuron.dendrites:
        if getattr(dend, "synapse_batch", None) is not None:
            rows.append(dend.synapse_batch.stdp_eligibility)
        else:
            rows.append(torch.stack([syn.stdp_eligibility for syn in dend.synapses]).flatten())
    return torch.stack(rows)


def test_toggles_off_preserve_forward_baseline():
    x = _make_input()
    cfg = _make_cfg(enable_phase_gating=False, enable_dynamic_channel_gating=False)

    torch.manual_seed(1)
    neuron_a = MPJRDNeuronAdvanced(cfg)
    torch.manual_seed(2)
    neuron_b = MPJRDNeuronAdvanced(cfg)
    neuron_b.load_state_dict(copy.deepcopy(neuron_a.state_dict()))

    out_a = neuron_a(x, collect_stats=True)
    out_b = neuron_b(x, collect_stats=True)

    assert torch.allclose(out_a["spikes"], out_b["spikes"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_a["u"], out_b["u"], atol=1e-6, rtol=1e-5)


def test_phase_gating_changes_stdp_updates_without_shape_break():
    x = _make_input()
    cfg_off = _make_cfg(enable_phase_gating=False)
    cfg_on = _make_cfg(enable_phase_gating=True)

    base = MPJRDNeuronAdvanced(cfg_off)
    exp = MPJRDNeuronAdvanced(cfg_on)
    exp.load_state_dict(copy.deepcopy(base.state_dict()))

    _ = base(x, collect_stats=False, phase=torch.zeros(x.shape[0]))
    out_on = exp(x, collect_stats=False, phase=torch.full((x.shape[0],), torch.pi))

    base_elig = _stdp_eligibility_tensor(base)
    exp_elig = _stdp_eligibility_tensor(exp)

    assert exp_elig.shape == base_elig.shape
    assert not torch.allclose(exp_elig, base_elig)
    assert out_on["spikes"].shape[0] == x.shape[0]


def test_config_and_state_dict_device_and_save_load_compatibility():
    cfg = _make_cfg(debug_collect_mechanism_metrics=True)
    toggles = MechanismToggleSet(cfg)
    assert toggles.is_enabled("phase_gating") is False

    neuron = MPJRDNeuronAdvanced(cfg)
    neuron.to(torch.device("cpu"))
    state = neuron.state_dict()

    restored = MPJRDNeuronAdvanced(cfg)
    restored.load_state_dict(copy.deepcopy(state))

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, map_location="cpu"))


def test_compare_mode_produces_ab_report_and_metrics():
    cfg_base = _make_cfg(enable_phase_gating=False)
    cfg_exp = _make_cfg(enable_phase_gating=True, debug_compare_baseline=True)
    x = _make_input()

    def _factory(enable: bool):
        cfg = cfg_exp if enable else cfg_base
        return MPJRDNeuronAdvanced(cfg)

    report = compare_mechanism_vs_baseline(
        factory=_factory,
        x=x,
        mechanism_name="phase_gating",
        forward_kwargs={"collect_stats": True, "phase": torch.full((x.shape[0],), 0.25)},
    )

    assert report.output_diff["mechanism_enabled"] == 1.0
    assert "spike_rate" in report.baseline_metrics
    assert "sparsity_ratio" in report.experiment_metrics


def test_ab_pattern_overlap_phase_gating_improves_weight_selectivity():
    x = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0], [1.0, 0.8, 0.0, 0.0]],
            [[1.0, 0.7, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        ]
    )
    cfg_base = _make_cfg(enable_phase_gating=False)
    cfg_exp = _make_cfg(enable_phase_gating=True)

    baseline = MPJRDNeuronAdvanced(cfg_base)
    experiment = MPJRDNeuronAdvanced(cfg_exp)
    experiment.load_state_dict(copy.deepcopy(baseline.state_dict()))

    _ = baseline(x, collect_stats=False, phase=torch.zeros(x.shape[0]))
    _ = experiment(x, collect_stats=False, phase=torch.full((x.shape[0],), torch.pi / 2))

    baseline_update = _stdp_eligibility_tensor(baseline).abs().mean()
    experiment_update = _stdp_eligibility_tensor(experiment).abs().mean()

    assert experiment_update < baseline_update
