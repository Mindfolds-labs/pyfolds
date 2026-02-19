import numpy as np
import pytest
import torch

from pyfolds.contracts import (
    CONTRACT_MECHANISM_ORDER,
    NeuronStepInput,
    TensorFlowNeuronContractBackend,
    TorchNeuronContractBackend,
)


def test_torch_contract_invariants_order_and_time_step_end_of_step():
    backend = TorchNeuronContractBackend()
    x = torch.tensor([[[1.0, 0.0], [0.2, 0.9]]], dtype=torch.float32)

    out = backend.run_step(NeuronStepInput(x=x, dt=1.5, time_step=10.0))

    assert tuple(out.step_trace.mechanism_order) == CONTRACT_MECHANISM_ORDER
    assert out.step_trace.time_step_before == 10.0
    assert out.step_trace.time_step_after == 11.5
    for step in CONTRACT_MECHANISM_ORDER:
        assert out.step_trace.mechanism_time_snapshot[step] == 10.0

    assert out.spikes.shape == (1,)
    assert out.somatic.shape == (1,)


def test_torch_and_tf_contract_conformance_with_same_artificial_input_and_tolerance():
    tf = pytest.importorskip("tensorflow")
    del tf

    x = np.array([[[1.0, 0.0], [0.2, 0.9]]], dtype=np.float32)
    step_input = NeuronStepInput(x=x, dt=1.0, time_step=7.0)

    torch_backend = TorchNeuronContractBackend()
    tf_backend = TensorFlowNeuronContractBackend()

    out_torch = torch_backend.run_step(step_input)
    out_tf = tf_backend.run_step(step_input)

    spikes_torch = out_torch.spikes.detach().cpu().numpy()
    somatic_torch = out_torch.somatic.detach().cpu().numpy()
    spikes_tf = out_tf.spikes.numpy()
    somatic_tf = out_tf.somatic.numpy()

    # Divergências aceitáveis entre backends (float32):
    atol = 1e-6
    rtol = 1e-5
    assert np.allclose(spikes_torch, spikes_tf, atol=atol, rtol=rtol)
    assert np.allclose(somatic_torch, somatic_tf, atol=atol, rtol=rtol)
