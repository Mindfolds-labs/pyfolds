import importlib.util

import pytest


def test_importing_pyfolds_still_works_without_tensorflow():
    import pyfolds

    assert hasattr(pyfolds, "MPJRDNeuron")


def test_tf_backend_guard_when_tensorflow_is_missing(monkeypatch):
    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("TensorFlow instalado; guard test não aplicável")

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    import pyfolds.tf as tf_backend

    with pytest.raises(ImportError, match="TensorFlow backend requested"):
        _ = tf_backend.MPJRDTFNeuronCell


def test_tf_cell_state_and_step_contract():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(units=3, threshold=0.5, decay=0.0)
    x = tf.ones((2, 3), dtype=tf.float32)
    state = cell.get_initial_state(batch_size=2, dtype=tf.float32)[0]

    spikes, next_state = cell.step(x, state, dt=1.0)

    assert spikes.shape == (2, 3)
    assert next_state.shape == (2, 3)


def test_tf_layer_integrates_with_keras_rnn():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFLayer

    layer = MPJRDTFLayer(units=4, return_sequences=True, return_state=True)
    x = tf.ones((2, 5, 4), dtype=tf.float32)

    out = layer(x)

    assert set(out.keys()) == {"spikes", "state"}
    assert out["spikes"].shape == (2, 5, 4)
    assert out["state"].shape == (2, 4)
