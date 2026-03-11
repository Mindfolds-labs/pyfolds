from __future__ import annotations

import numpy as np
import pytest

from pyfolds.contracts import NeuronStepInput, TensorFlowNeuronContractBackend, TorchNeuronContractBackend


_LEVEL_ORDER = {"A": 3, "B": 2, "C": 1}
_LEVEL_TOLERANCES = {
    "A": (1e-6, 1e-5),
    "B": (1e-4, 1e-3),
    "C": (1e-2, 1e-1),
}


def _achieved_level(a: np.ndarray, b: np.ndarray) -> str:
    abs_err = float(np.max(np.abs(a - b)))
    denom = np.maximum(np.abs(b), 1e-12)
    rel_err = float(np.max(np.abs(a - b) / denom))
    for level in ("A", "B", "C"):
        atol, rtol = _LEVEL_TOLERANCES[level]
        if abs_err <= atol and rel_err <= rtol:
            return level
    return "below-C"


def _assert_declared_level_not_overstated(*, component: str, declared: str, achieved: str) -> None:
    assert achieved in _LEVEL_ORDER, (
        f"{component} ficou abaixo de C (achieved={achieved})."
    )
    assert _LEVEL_ORDER[achieved] >= _LEVEL_ORDER[declared], (
        f"{component} declara nível {declared}, mas entrega apenas {achieved}."
    )


@pytest.mark.tf
@pytest.mark.parametrize("target_level", ["A", "B", "C"])
def test_pt_vs_tf_shape_dtype_masking_and_equivalence_by_level(target_level: str):
    pytest.importorskip("tensorflow")

    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, size=(3, 2, 4)).astype(np.float32)
    mask = np.array(
        [
            [[1, 1, 1, 0], [1, 0, 1, 1]],
            [[0, 1, 1, 1], [1, 1, 0, 1]],
            [[1, 0, 0, 1], [1, 1, 1, 1]],
        ],
        dtype=np.float32,
    )
    x_masked = x * mask

    step_input = NeuronStepInput(x=x_masked, dt=1.0, time_step=2.0)

    torch_backend = TorchNeuronContractBackend()
    tf_backend = TensorFlowNeuronContractBackend()

    out_pt = torch_backend.run_step(step_input)
    out_tf = tf_backend.run_step(step_input)

    spikes_pt = out_pt.spikes.detach().cpu().numpy()
    spikes_tf = out_tf.spikes.numpy()
    somatic_pt = out_pt.somatic.detach().cpu().numpy()
    somatic_tf = out_tf.somatic.numpy()

    assert spikes_pt.shape == spikes_tf.shape == (3,)
    assert somatic_pt.shape == somatic_tf.shape == (3,)
    assert spikes_pt.dtype == spikes_tf.dtype == np.float32
    assert somatic_pt.dtype == somatic_tf.dtype == np.float32

    atol, rtol = _LEVEL_TOLERANCES[target_level]
    assert np.allclose(spikes_pt, spikes_tf, atol=atol, rtol=rtol)
    assert np.allclose(somatic_pt, somatic_tf, atol=atol, rtol=rtol)

    # Máscara precisa ter efeito perceptível nos dois backends.
    raw_step = NeuronStepInput(x=x, dt=1.0, time_step=2.0)
    out_pt_raw = TorchNeuronContractBackend().run_step(raw_step)
    out_tf_raw = TensorFlowNeuronContractBackend().run_step(raw_step)
    raw_somatic_pt = out_pt_raw.somatic.detach().cpu().numpy()
    raw_somatic_tf = out_tf_raw.somatic.numpy()

    assert np.any(np.abs(raw_somatic_pt - somatic_pt) > 1e-6)
    assert np.any(np.abs(raw_somatic_tf - somatic_tf) > 1e-6)

    achieved = _achieved_level(somatic_pt, somatic_tf)
    _assert_declared_level_not_overstated(
        component="torch_vs_tf_contract_backend",
        declared="A",
        achieved=achieved,
    )


@pytest.mark.tf
def test_tf_wrapper_saturation_compression_and_local_threshold_semantics():
    tf = pytest.importorskip("tensorflow")

    from pyfolds.tf import MPJRDTFNeuronCell

    x = tf.constant([[4.0, 2.0]], dtype=tf.float32)
    prev = tf.zeros((1, 2), dtype=tf.float32)

    low_thr = MPJRDTFNeuronCell(units=2, threshold=0.5, decay=0.0)
    high_thr = MPJRDTFNeuronCell(units=2, threshold=3.0, decay=0.0)

    spikes_low, state_low = low_thr.step(x, prev, dt=1.0)
    spikes_high, state_high = high_thr.step(x, prev, dt=1.0)

    spikes_low_np = spikes_low.numpy()
    spikes_high_np = spikes_high.numpy()
    state_low_np = state_low.numpy()
    state_high_np = state_high.numpy()

    # Semântica de threshold local: threshold maior não pode gerar mais spikes.
    assert np.all(spikes_high_np <= spikes_low_np)

    # Semântica aproximada de compressão/saturação: estado pós-spike mantém resíduo limitado.
    assert np.all(state_low_np < 4.0)
    assert np.all(state_high_np < 4.0)

    expected_spikes_low = (x.numpy() >= 0.5).astype(np.float32)
    expected_state_low = x.numpy() - (expected_spikes_low * 0.5)
    achieved = _achieved_level(state_low_np, expected_state_low)
    _assert_declared_level_not_overstated(
        component="tf_wrapper_threshold_semantics",
        declared="A",
        achieved=achieved,
    )

    assert np.allclose(spikes_low_np, expected_spikes_low, atol=1e-6, rtol=1e-5)
