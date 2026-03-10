import time

import torch

from pyfolds.integration import CognitiveBatch, OptimizedMPJRDNeuron


class _MockNoeticBridge:
    def process_text(self, text_batch: list[str]) -> CognitiveBatch:
        batch = len(text_batch)
        seq_len = 12
        concept_dim = 16
        embeddings = torch.randn(batch, seq_len, concept_dim)
        confidence = torch.sigmoid(torch.randn(batch, seq_len))
        surprise = torch.sigmoid(torch.randn(batch, seq_len))
        return CognitiveBatch(
            concept_embeddings=embeddings,
            confidence=confidence,
            surprise=surprise,
        )

    def consume_feedback(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        return payload["spikes"].mean()


class _BaselineLoopNeuron(torch.nn.Module):
    def __init__(self, dendrites: int, synapses: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dendrites, synapses) * 0.05)
        self.threshold = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for b in range(x.shape[0]):
            dend_sum = 0.0
            for d in range(x.shape[1]):
                dend_sum += torch.sum(x[b, d] * self.weight[d])
            out.append((dend_sum > self.threshold).to(x.dtype))
        return torch.stack(out, dim=0)


def test_noetic_to_pyfolds_pipeline_shapes_and_batch() -> None:
    bridge = _MockNoeticBridge()
    cognitive = bridge.process_text(["alpha", "beta", "gamma", "delta"])
    cognitive.validate()

    neuron = OptimizedMPJRDNeuron(dendrites=12, synapses=16, hidden_dim=24)
    output = neuron(
        cognitive.concept_embeddings,
        confidence=cognitive.confidence,
        surprise=cognitive.surprise,
        apply_stdp=True,
    )
    noetic_payload = output.to_noetic()
    score = bridge.consume_feedback(noetic_payload)

    assert output.spikes.shape == (4,)
    assert output.membrane_potential.shape == (4,)
    assert output.dendritic_states.shape == (4, 24)
    assert output.cognitive_feedback is not None
    assert output.cognitive_feedback.shape == (4,)
    assert score.ndim == 0


def test_latency_under_100ms_and_perf_gain() -> None:
    torch.manual_seed(7)
    batch, dendrites, synapses = 64, 24, 24
    hidden = 64
    x = torch.randn(batch, dendrites, synapses)

    optimized = OptimizedMPJRDNeuron(dendrites=dendrites, synapses=synapses, hidden_dim=hidden)
    baseline = _BaselineLoopNeuron(dendrites=dendrites, synapses=synapses)

    for _ in range(5):
        _ = optimized(x)
        _ = baseline(x)

    t0 = time.perf_counter()
    for _ in range(50):
        _ = optimized(x)
    optimized_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    for _ in range(50):
        _ = baseline(x)
    baseline_ms = (time.perf_counter() - t1) * 1000.0

    assert optimized_ms < 100.0
    assert baseline_ms / optimized_ms > 1.5
