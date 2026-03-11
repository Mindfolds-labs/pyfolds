import torch
import pyfolds


def test_strict_contract_rejects_wta_hard_path() -> None:
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        dendrite_integration_mode="wta_hard",
        contract_enforcement="strict",
    )
    neuron = pyfolds.MPJRDNeuron(cfg)
    x = torch.rand(2, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    try:
        neuron(x, collect_stats=False)
        raised = False
    except RuntimeError:
        raised = True

    assert raised


def test_audit_trace_buffer_updates_in_light_mode() -> None:
    cfg = pyfolds.NeuronConfig(
        n_dendrites=2,
        n_synapses_per_dendrite=4,
        audit_mode="light",
        audit_trace_capacity=8,
        contract_enforcement="off",
    )
    neuron = pyfolds.MPJRDNeuron(cfg)
    x = torch.rand(1, cfg.n_dendrites, cfg.n_synapses_per_dendrite)

    for _ in range(3):
        neuron(x, collect_stats=False)

    snap = neuron.get_audit_trace_snapshot()
    assert snap["winner_idx"].numel() == 8
    assert int(snap["pointer"].item()) == 3


def test_legacy_config_access_for_audit_fields() -> None:
    cfg = pyfolds.NeuronConfig(audit_mode="full", contract_enforcement="warn")
    assert cfg.audit.audit_mode == "full"
    assert cfg.contract_enforcement == "warn"
