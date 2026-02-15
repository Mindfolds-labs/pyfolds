from pathlib import Path

import pyfolds

from pyfolds.monitoring import HealthStatus, NeuronHealthCheck
from pyfolds.serialization import VersionedCheckpoint


class DummyNeuron:
    def __init__(self, metrics):
        self._metrics = metrics

    def get_metrics(self):
        return self._metrics


def test_health_check_critical_for_dead_neurons():
    checker = NeuronHealthCheck(DummyNeuron({"dead_neuron_ratio": 0.2}))
    status, alerts = checker.check()

    assert status == HealthStatus.CRITICAL
    assert alerts


def test_versioned_checkpoint_save_and_load(tmp_path):
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    neuron = pyfolds.MPJRDNeuron(cfg)
    ckpt = VersionedCheckpoint(neuron, version="1.0.0")

    path = Path(tmp_path) / "neuron.pt"
    payload = ckpt.save(str(path), extra_metadata={"experiment": "unit"})
    loaded = VersionedCheckpoint.load(str(path), model=neuron)

    assert payload["integrity_hash"] == loaded["integrity_hash"]
    assert loaded["metadata"]["version"] == "1.0.0"
    assert loaded["metadata"]["experiment"] == "unit"
