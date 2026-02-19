from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Sequence


class MechanismStep(str, Enum):
    STP = "stp"
    INTEGRATION = "integration"
    SFA = "sfa"
    THRESHOLD = "threshold"
    REFRACTORY = "refractory"
    BAP = "bap"
    STDP = "stdp"
    HOMEOSTASIS = "homeostasis"


CONTRACT_MECHANISM_ORDER = (
    MechanismStep.STP,
    MechanismStep.INTEGRATION,
    MechanismStep.SFA,
    MechanismStep.THRESHOLD,
    MechanismStep.REFRACTORY,
    MechanismStep.BAP,
    MechanismStep.STDP,
    MechanismStep.HOMEOSTASIS,
)


@dataclass(frozen=True)
class NeuronStepInput:
    x: object
    dt: float
    time_step: float


@dataclass(frozen=True)
class StepExecutionTrace:
    mechanism_order: Sequence[MechanismStep]
    mechanism_time_snapshot: Dict[MechanismStep, float]
    time_step_before: float
    time_step_after: float


@dataclass(frozen=True)
class NeuronStepOutput:
    spikes: object
    somatic: object
    step_trace: StepExecutionTrace


class ContractViolation(ValueError):
    pass


def validate_step_output(output: NeuronStepOutput, dt: float) -> None:
    trace = output.step_trace
    if tuple(trace.mechanism_order) != CONTRACT_MECHANISM_ORDER:
        raise ContractViolation(
            "Ordem de mecanismos inválida: "
            f"esperado={[s.value for s in CONTRACT_MECHANISM_ORDER]} "
            f"recebido={[s.value for s in trace.mechanism_order]}"
        )

    for step in CONTRACT_MECHANISM_ORDER:
        snapshot = trace.mechanism_time_snapshot.get(step)
        if snapshot is None:
            raise ContractViolation(f"Snapshot de tempo ausente para mecanismo {step.value}")
        if snapshot != trace.time_step_before:
            raise ContractViolation(
                f"time_step mudou antes do fim do passo no mecanismo {step.value}: "
                f"{snapshot} != {trace.time_step_before}"
            )

    expected_after = trace.time_step_before + dt
    if trace.time_step_after != expected_after:
        raise ContractViolation(
            f"time_step final inválido: esperado {expected_after}, recebido {trace.time_step_after}"
        )
