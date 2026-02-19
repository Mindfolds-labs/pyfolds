from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Sequence


class MechanismStep(str, Enum):
    """Enum com a ordem canônica dos mecanismos executados em um passo neuronal."""

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
    """Entrada mínima necessária para executar um passo (`step`) do neurônio."""

    x: object
    dt: float
    time_step: float


@dataclass(frozen=True)
class StepExecutionTrace:
    """Rastro de execução do passo contendo ordem e snapshots temporais."""

    mechanism_order: Sequence[MechanismStep]
    mechanism_time_snapshot: Dict[MechanismStep, float]
    time_step_before: float
    time_step_after: float


@dataclass(frozen=True)
class NeuronStepOutput:
    """Saída contratual de um passo com spikes, potencial somático e rastro."""

    spikes: object
    somatic: object
    step_trace: StepExecutionTrace


class ContractViolation(ValueError):
    """Erro lançado quando a saída de `step` viola o contrato formal."""

    pass


def validate_step_output(output: NeuronStepOutput, dt: float) -> None:
    """Valida invariantes de ordem e avanço temporal da execução de `step`."""

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
