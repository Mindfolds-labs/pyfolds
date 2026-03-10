from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .types import PyFoldsOutput


class SurrogateGradientFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: Tensor, threshold: Tensor, beta: float) -> Tensor:
        centered = membrane - threshold
        ctx.save_for_backward(centered)
        ctx.beta = beta
        return (centered > 0).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (centered,) = ctx.saved_tensors
        beta = ctx.beta
        sig = torch.sigmoid(centered)
        grad = beta * sig * (1.0 - sig)
        grad_input = grad_output * grad
        grad_threshold = -grad_input
        return grad_input, grad_threshold, None


@dataclass
class STDPState:
    pre_trace: Tensor
    post_trace: Tensor


class OptimizedMPJRDNeuron(nn.Module):
    """Vectorized MPJRD neuron core with optional STDP and cognitive feedback."""

    def __init__(
        self,
        dendrites: int,
        synapses: int,
        hidden_dim: int,
        beta: float = 5.0,
        threshold: float = 0.2,
        stdp_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dendrites = dendrites
        self.synapses = synapses
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.stdp_lr = stdp_lr

        self.synaptic_kernel = nn.Parameter(torch.randn(dendrites, synapses) * 0.05)
        self.dendritic_gate = nn.Parameter(torch.randn(dendrites, hidden_dim) * 0.05)
        self.soma_projection = nn.Parameter(torch.randn(hidden_dim) * 0.05)
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def _update_stdp(self, pre_spikes: Tensor, post_spikes: Tensor) -> None:
        # pre_spikes: [batch, dendrites, synapses], post_spikes: [batch]
        pre_activity = pre_spikes.mean(dim=0)
        post_activity = post_spikes.unsqueeze(-1).unsqueeze(-1)
        post_activity = post_activity.mean(dim=0)
        hebb = pre_activity * post_activity
        anti_hebb = pre_activity * (1.0 - post_activity)
        delta = self.stdp_lr * (hebb - anti_hebb)
        with torch.no_grad():
            self.synaptic_kernel.add_(delta)

    def forward(
        self,
        x: Tensor,
        confidence: Optional[Tensor] = None,
        surprise: Optional[Tensor] = None,
        apply_stdp: bool = False,
    ) -> PyFoldsOutput:
        if x.ndim != 3:
            raise ValueError("Expected x with shape (batch, dendrites, synapses)")
        if x.shape[1] != self.dendrites or x.shape[2] != self.synapses:
            raise ValueError("Input dimensions do not match configured dendrites/synapses")

        # 1) synaptic projection [batch, dendrites]
        synaptic_projection = torch.einsum("bds,ds->bd", x, self.synaptic_kernel)

        # 2) dendritic activation [batch, hidden_dim]
        dendritic_states = torch.tanh(torch.einsum("bd,dh->bh", synaptic_projection, self.dendritic_gate))

        # 3) spike generation [batch]
        membrane = torch.einsum("bh,h->b", dendritic_states, self.soma_projection)
        spikes = SurrogateGradientFn.apply(membrane, self.threshold, self.beta)

        # 4) soma integration [batch]
        membrane_potential = membrane + 0.1 * spikes

        # 5) optional cognitive feedback [batch]
        cognitive_feedback = None
        if confidence is not None and surprise is not None:
            if confidence.shape != surprise.shape:
                raise ValueError("confidence and surprise must share the same shape")
            if confidence.ndim > 1:
                confidence = confidence.mean(dim=-1)
                surprise = surprise.mean(dim=-1)
            cognitive_feedback = spikes * confidence - (1.0 - spikes) * surprise

        if apply_stdp:
            self._update_stdp(x.detach(), spikes.detach())

        return PyFoldsOutput(
            spikes=spikes,
            membrane_potential=membrane_potential,
            dendritic_states=dendritic_states,
            cognitive_feedback=cognitive_feedback,
        )
