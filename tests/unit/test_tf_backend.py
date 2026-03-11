import importlib
import importlib.util
import sys

import pytest


def test_importing_pyfolds_still_works_without_tensorflow():
    import pyfolds

    assert hasattr(pyfolds, "MPJRDNeuron")


def test_tf_backend_guard_when_tensorflow_is_missing(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    sys.modules.pop("pyfolds.tf", None)

    import pyfolds.tf as tf_backend
    importlib.reload(tf_backend)

    with pytest.raises(ImportError, match="TensorFlow backend requested"):
        _ = tf_backend.MPJRDTFNeuronCell

    sys.modules.pop("pyfolds.tf", None)


def test_tf_cell_state_and_step_contract():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(units=3, threshold=0.5, decay=0.0)
    x = tf.ones((2, 3), dtype=tf.float32)
    state = cell.get_initial_state(batch_size=2, dtype=tf.float32)

    spikes, next_state, adapt_state, context_state = cell.step(x, *state, dt=1.0)

    assert spikes.shape == (2, 3)
    assert next_state.shape == (2, 3)
    assert adapt_state.shape == (2, 3)
    assert context_state.shape == (2, 1)


def test_tf_cell_rejects_non_positive_dt():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(units=2)
    x = tf.ones((1, 2), dtype=tf.float32)
    state = cell.get_initial_state(batch_size=1, dtype=tf.float32)

    with pytest.raises(ValueError, match="Invalid argument `dt`"):
        cell.step(x, *state, dt=0.0)


def test_tf_layer_rejects_invalid_units():
    pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFLayer

    with pytest.raises(ValueError, match="Invalid argument `units`"):
        MPJRDTFLayer(units=0)


def test_tf_layer_integrates_with_keras_rnn():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFLayer

    layer = MPJRDTFLayer(units=4, return_sequences=True, return_state=True)
    x = tf.ones((2, 5, 4), dtype=tf.float32)

    out = layer(x)

    assert set(out.keys()) == {"spikes", "state"}
    assert set(out["state"].keys()) == {"membrane", "adapt_state", "context_state"}
    assert out["spikes"].shape == (2, 5, 4)
    assert out["state"]["membrane"].shape == (2, 4)
    assert out["state"]["adapt_state"].shape == (2, 4)
    assert out["state"]["context_state"].shape == (2, 1)


def test_tf_cell_structured_dendritic_input_and_diagnostics():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(units=3, threshold=0.5, decay=0.0, clip_value=5.0)
    x = tf.ones((2, 4, 5), dtype=tf.float32)
    state = cell.get_initial_state(batch_size=2, dtype=tf.float32)

    spikes, next_state, adapt_state, context_state, diagnostics = cell.step(
        x,
        *state,
        dt=1.0,
        return_diagnostics=True,
    )

    assert spikes.shape == (2, 3)
    assert next_state.shape == (2, 3)
    assert adapt_state.shape == (2, 3)
    assert context_state.shape == (2, 1)
    assert diagnostics["u"].shape == (2, 3)
    assert diagnostics["theta_eff"].shape == (2, 3)
    assert diagnostics["v_dend"].shape == (2, 4)
    assert diagnostics["adapt_state"].shape == (2, 3)
    assert diagnostics["context_state"].shape == (2, 1)


def test_tf_cell_applies_connectivity_and_pruning_masks():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    x = tf.constant([[[1.0, 1.0], [1.0, 1.0]]], dtype=tf.float32)
    connectivity_mask = tf.constant([[[1.0, 0.0], [1.0, 0.0]]], dtype=tf.float32)
    pruning_mask = tf.constant([[[1.0, 1.0], [0.0, 0.0]]], dtype=tf.float32)

    cell = MPJRDTFNeuronCell(
        units=2,
        threshold=100.0,
        decay=0.0,
        connectivity_mask=connectivity_mask,
        pruning_mask=pruning_mask,
    )
    state = cell.get_initial_state(batch_size=1, dtype=tf.float32)

    _, _, _, _, diagnostics = cell.step(x, *state, return_diagnostics=True)
    v_dend = diagnostics["v_dend"].numpy()

    assert v_dend.shape == (1, 2)
    assert v_dend[0, 0] == pytest.approx(1.0)
    assert v_dend[0, 1] == pytest.approx(0.0)


def test_tf_cell_silently_blocks_unsupported_mode_and_edge_plasticity():
    pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(
        units=2,
        integration_mode="sum",
        edge_inference_mode=True,
        plasticity_mode="stdp",
    )

    assert cell.integration_mode == "wta_soft_approx"
    assert cell.requested_integration_mode == "sum"
    assert cell.plasticity_mode == "none"


def test_tf_cell_clips_intermediates():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    cell = MPJRDTFNeuronCell(units=2, threshold=100.0, decay=0.0, clip_value=0.25)
    x = tf.constant([[10.0, -10.0]], dtype=tf.float32)
    state = cell.get_initial_state(batch_size=1, dtype=tf.float32)

    _, next_state, _, _, diagnostics = cell.step(x, *state, return_diagnostics=True)
    u = diagnostics["u"].numpy()
    v = next_state.numpy()

    assert u.max() <= 0.25 and u.min() >= -0.25
    assert v.max() <= 0.25 and v.min() >= -0.25
