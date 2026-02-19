from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .neuron_contract import (
    CONTRACT_MECHANISM_ORDER,
    MechanismStep,
    NeuronStepInput,
    NeuronStepOutput,
    StepExecutionTrace,
    validate_step_output,
)

try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - ambiente opcional
    tf = None


@dataclass
class _State:
    stp_u: float = 0.2
    stp_r: float = 1.0
    adaptation: float = 0.0
    refractory_until: float = -1_000.0


class _BaseContractBackend:
    def __init__(self) -> None:
        self.state = _State()

    def _trace(self, t0: float) -> Dict[MechanismStep, float]:
        return {step: t0 for step in CONTRACT_MECHANISM_ORDER}


class TorchNeuronContractBackend(_BaseContractBackend):
    """Implementação de referência do contrato com torch."""

    def run_step(self, step_input: NeuronStepInput) -> NeuronStepOutput:
        x = torch.as_tensor(step_input.x, dtype=torch.float32)
        if x.dim() != 3:
            raise ValueError("x deve ter shape [B, D, S]")

        t0 = float(step_input.time_step)
        dt = float(step_input.dt)

        # STP
        pre = (x > 0.5).float().mean().item()
        self.state.stp_u = float(np.clip(self.state.stp_u * 0.95 + 0.2 * pre, 0.0, 1.0))
        self.state.stp_r = float(np.clip(self.state.stp_r * 0.98 - self.state.stp_u * pre * 0.1, 0.0, 1.0))
        x_mod = x * (self.state.stp_u * self.state.stp_r)

        # Integração
        somatic = x_mod.sum(dim=(1, 2))

        # SFA
        self.state.adaptation = float(min(2.0, self.state.adaptation * 0.9))
        somatic_eff = somatic - self.state.adaptation

        # Threshold
        spikes = (somatic_eff >= 0.8).float()

        # Refratário
        if t0 < self.state.refractory_until:
            spikes = torch.zeros_like(spikes)

        # bAP
        if spikes.any():
            self.state.refractory_until = t0 + 2.0

        # STDP + Homeostase (sem alterar saída neste backend mínimo)

        trace = StepExecutionTrace(
            mechanism_order=list(CONTRACT_MECHANISM_ORDER),
            mechanism_time_snapshot=self._trace(t0),
            time_step_before=t0,
            time_step_after=t0 + dt,
        )
        output = NeuronStepOutput(spikes=spikes, somatic=somatic_eff, step_trace=trace)
        validate_step_output(output, dt=dt)
        return output


class TensorFlowNeuronContractBackend(_BaseContractBackend):
    """Implementação equivalente usando TensorFlow (quando disponível)."""

    def run_step(self, step_input: NeuronStepInput) -> NeuronStepOutput:
        if tf is None:
            raise RuntimeError("TensorFlow não está instalado no ambiente")

        x = tf.convert_to_tensor(step_input.x, dtype=tf.float32)
        if len(x.shape) != 3:
            raise ValueError("x deve ter shape [B, D, S]")

        t0 = float(step_input.time_step)
        dt = float(step_input.dt)

        pre = float(tf.reduce_mean(tf.cast(x > 0.5, tf.float32)).numpy())
        self.state.stp_u = float(np.clip(self.state.stp_u * 0.95 + 0.2 * pre, 0.0, 1.0))
        self.state.stp_r = float(np.clip(self.state.stp_r * 0.98 - self.state.stp_u * pre * 0.1, 0.0, 1.0))
        x_mod = x * (self.state.stp_u * self.state.stp_r)

        somatic = tf.reduce_sum(x_mod, axis=(1, 2))
        self.state.adaptation = float(min(2.0, self.state.adaptation * 0.9))
        somatic_eff = somatic - self.state.adaptation
        spikes = tf.cast(somatic_eff >= 0.8, tf.float32)

        if t0 < self.state.refractory_until:
            spikes = tf.zeros_like(spikes)

        if bool(tf.reduce_any(spikes > 0).numpy()):
            self.state.refractory_until = t0 + 2.0

        trace = StepExecutionTrace(
            mechanism_order=list(CONTRACT_MECHANISM_ORDER),
            mechanism_time_snapshot=self._trace(t0),
            time_step_before=t0,
            time_step_after=t0 + dt,
        )

        output = NeuronStepOutput(spikes=spikes, somatic=somatic_eff, step_trace=trace)
        validate_step_output(output, dt=dt)
        return output
