"""Integration tests for MindControl runtime parameter mutation."""

import pytest
import torch
import torch.nn as nn

from pyfolds.monitoring.mindcontrol import MindControlEngine, MutationQueue


class MockNeuron(nn.Module):
    """Minimal neuron to validate graph-safe runtime mutation."""

    def __init__(self, neuron_id: str):
        super().__init__()
        self.neuron_id = neuron_id
        self.activity_threshold = nn.Parameter(torch.tensor(0.5))
        self._mutation_queue: MutationQueue | None = None

    def attach_mindcontrol(self, mq: MutationQueue) -> None:
        self._mutation_queue = mq

    def _apply_mutations_safe(self) -> None:
        if self._mutation_queue is None:
            return

        for param_name, new_val in self._mutation_queue.fetch_all():
            target = getattr(self, param_name, None)
            if target is None:
                continue
            with torch.no_grad():
                target.copy_(torch.tensor(new_val, dtype=target.dtype, device=target.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._apply_mutations_safe()
        return x * self.activity_threshold


@pytest.mark.integration
def test_mindcontrol_graph_safety() -> None:
    """Runtime mutation must not break PyTorch autograd graph lifecycle."""
    engine = MindControlEngine()
    neuron = MockNeuron(neuron_id="test_n1")
    mq = engine.register_neuron("test_n1")
    neuron.attach_mindcontrol(mq)

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    loss_1 = neuron(x).sum()
    loss_1.backward()

    engine.analyze_and_react({"neuron_id": "test_n1", "spike_rate": 0.01})

    loss_2 = neuron(x).sum()
    assert neuron.activity_threshold.item() == pytest.approx(0.1)

    try:
        loss_2.backward()
    except RuntimeError as exc:
        pytest.fail(f"MindControl mutation broke autograd graph: {exc}")


@pytest.mark.integration
def test_mindcontrol_bounds_clamp_threshold_values() -> None:
    """Safety bounds must clamp engineered thresholds to safe runtime values."""
    engine = MindControlEngine()
    queue = engine.register_neuron("n-safe")

    engine.safety_bounds["activity_threshold"] = (0.2, 0.4)
    engine.analyze_and_react({"neuron_id": "n-safe", "spike_rate": 0.0})

    assert queue.fetch_all() == [("activity_threshold", 0.2)]
