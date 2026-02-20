from datetime import datetime
import json
from pathlib import Path

import pytest
import pyfolds
import torch

from pyfolds.monitoring import HealthStatus, ModelIntegrityMonitor, NeuronHealthCheck, WeightIntegrityMonitor
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



def test_health_check_uses_fallback_metrics_from_get_metrics_contract():
    checker = NeuronHealthCheck(
        DummyNeuron({"protection_ratio": 0.4, "r_hat": 0.0}),
        thresholds={
            "dead_neuron_rate": 0.05,
            "saturation_ratio": 0.30,
            "min_spike_rate": 0.01,
        },
    )
    status, alerts = checker.check()

    assert status == HealthStatus.CRITICAL
    assert any("neurônios mortos" in alert for alert in alerts)

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

def test_versioned_checkpoint_metadata_created_at_is_utc(tmp_path):
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    neuron = pyfolds.MPJRDNeuron(cfg)
    ckpt = VersionedCheckpoint(neuron, version="1.0.0")

    path = Path(tmp_path) / "neuron-metadata.pt"
    payload = ckpt.save(str(path))

    created_at = payload["metadata"]["created_at"]
    assert created_at.endswith("Z")
    assert datetime.fromisoformat(created_at.replace("Z", "+00:00")).tzinfo is not None



def test_versioned_checkpoint_shape_validation_raises_on_mismatch(tmp_path):
    cfg_source = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    cfg_target = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=5)
    source = pyfolds.MPJRDNeuron(cfg_source)
    target = pyfolds.MPJRDNeuron(cfg_target)

    ckpt = VersionedCheckpoint(source, version="1.0.0")
    path = Path(tmp_path) / "neuron-shape.pt"
    ckpt.save(str(path))

    try:
        VersionedCheckpoint.load(str(path), model=target)
        assert False, "esperava ValueError por shape mismatch"
    except ValueError as exc:
        assert "incompatível" in str(exc) or "Shape mismatch" in str(exc)


def test_versioned_checkpoint_safetensors_roundtrip(tmp_path):
    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    neuron = pyfolds.MPJRDNeuron(cfg)
    ckpt = VersionedCheckpoint(neuron, version="1.0.0")

    path = Path(tmp_path) / "neuron.pt"

    try:
        payload = ckpt.save(str(path), use_safetensors=True, extra_metadata={"backend": "safe"})
    except RuntimeError as exc:
        if "safetensors" in str(exc):
            return
        raise

    safetensor_path = path.with_suffix('.safetensors')
    assert safetensor_path.exists()
    assert safetensor_path.with_suffix('.safetensors.meta.json').exists()

    loaded = VersionedCheckpoint.load(str(safetensor_path), model=neuron)
    assert loaded["metadata"]["backend"] == "safe"
    assert loaded["integrity_hash"] == payload["integrity_hash"]


def test_model_integrity_monitor_detects_unexpected_mutation():
    model = pyfolds.MPJRDNeuron(pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4))
    monitor = ModelIntegrityMonitor(model, check_every_n_steps=1)

    baseline = monitor.set_baseline()
    first = monitor.check_integrity()

    assert first["integrity_ok"] is True
    assert first["expected_hash"] == baseline

    with torch.no_grad():
        buf = next(model.buffers())
        buf.add_(1)

    second = monitor.check_integrity()
    assert second["integrity_ok"] is False
    assert second["current_hash"] != second["expected_hash"]


def test_model_integrity_monitor_initializes_hash_on_first_check():
    model = pyfolds.MPJRDNeuron(pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4))
    monitor = ModelIntegrityMonitor(model, check_every_n_steps=1)

    payload = monitor.check_integrity()

    assert payload["hash_initialized"] is True
    assert isinstance(payload["current_hash"], str)
    assert len(payload["current_hash"]) == 64


def test_weight_integrity_monitor_detects_mutation_between_checks():
    model = pyfolds.MPJRDNeuron(pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4))
    monitor = WeightIntegrityMonitor(model, check_every_n_steps=1)

    first = monitor.check()
    assert first["checked"] is True
    assert first["ok"] is True

    with torch.no_grad():
        buf = next(model.buffers())
        buf.add_(1)

    second = monitor.check()
    assert second["checked"] is True
    assert second["ok"] is False


def test_versioned_checkpoint_load_secure_validates_hash_and_shapes(tmp_path):
    safetensors = pytest.importorskip("safetensors")

    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    model = pyfolds.MPJRDNeuron(cfg)
    ckpt = VersionedCheckpoint(model, version="2.0.3")

    state = model.state_dict()
    weights_path = tmp_path / "secure.safetensors"
    safetensors.torch.save_file(state, str(weights_path))

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "weight_file": weights_path.name,
        "metadata": {"version": "2.0.3"},
        "integrity_hash": ckpt._compute_hash(state),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    metadata = VersionedCheckpoint.load_secure(str(manifest_path), model)
    assert metadata["version"] == "2.0.3"


def test_versioned_checkpoint_load_secure_fails_on_hash_mismatch(tmp_path):
    safetensors = pytest.importorskip("safetensors")

    cfg = pyfolds.MPJRDConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    model = pyfolds.MPJRDNeuron(cfg)
    state = model.state_dict()

    weights_path = tmp_path / "corrupted.safetensors"
    safetensors.torch.save_file(state, str(weights_path))

    manifest_path = tmp_path / "manifest-bad.json"
    manifest = {
        "weight_file": weights_path.name,
        "metadata": {"version": "2.0.3"},
        "integrity_hash": "sha256:" + "0" * 64,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="integridade"):
        VersionedCheckpoint.load_secure(str(manifest_path), model)
