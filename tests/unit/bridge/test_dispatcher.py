from pyfolds.bridge import MindDispatcher
import torch


class _Cfg:
    def to_dict(self):
        return {"a": 1}


class _Layer:
    cfg = _Cfg()


class _Network:
    layers = [_Layer()]


def test_prepare_payload_tensor_fields_are_serializable():
    payload = MindDispatcher.prepare_payload(
        layer_id="L0",
        spikes=torch.tensor([1.0, 0.0]),
        weights=torch.tensor([1.0, 3.0]),
        health_score=0.9,
    )

    assert payload["layer_id"] == "L0"
    assert isinstance(payload["timestamp"], str)
    assert payload["data"]["spikes"] == [1.0, 0.0]
    assert payload["data"]["weights_mean"] == 2.0
    assert payload["data"]["health"] == 0.9


def test_get_topology_map_uses_cfg_to_dict_when_available():
    topology = MindDispatcher.get_topology_map(_Network())
    assert topology == [{"layer_index": 0, "config": {"a": 1}}]


def test_capture_event_preserves_new_and_legacy_contract_keys():
    event = MindDispatcher.capture_event(
        layer_id="L1",
        spikes=torch.tensor([0.0, 1.0]),
        weights=torch.tensor([2.0, 6.0]),
        metrics={"health": 1.0},
    )

    assert event["layer"] == "L1"
    assert isinstance(event["ts"], str)
    assert event["layer_id"] == "L1"
    assert event["timestamp"] == event["ts"]

