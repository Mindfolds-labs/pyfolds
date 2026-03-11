"""Runtime enforcement for scientific execution contracts in MPJRD."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ScientificStage(str, Enum):
    """Canonical stages for dendritic-to-somatic processing."""

    LOCAL_NONLINEARITY = "local_nonlinearity"
    COMPETITION = "competition"
    GLOBAL_AGGREGATION = "global_aggregation"


EXPECTED_DENDRITIC_ORDER: tuple[ScientificStage, ...] = (
    ScientificStage.LOCAL_NONLINEARITY,
    ScientificStage.COMPETITION,
    ScientificStage.GLOBAL_AGGREGATION,
)


@dataclass(frozen=True)
class ScientificContract:
    """Contract payload to validate runtime stage order.

    Parameters
    ----------
    stage_order : Sequence[ScientificStage]
        Order observed during current ``forward`` execution.
    """

    stage_order: Sequence[ScientificStage]


class ScientificContractViolation(RuntimeError):
    """Raised when scientific execution order contract is violated."""


class ContractEnforcer:
    """Validates scientific contracts using configurable enforcement levels.

    Parameters
    ----------
    level : str
        Enforcement level: ``off``, ``warn`` or ``strict``.
    """

    def __init__(self, level: str = "warn") -> None:
        self.level = level

    def validate(self, contract: ScientificContract) -> None:
        """Validate stage ordering according to configured level.

        Parameters
        ----------
        contract : ScientificContract
            Runtime contract with observed stage order.
        """
        if self.level == "off":
            return

        observed = tuple(contract.stage_order)
        if observed == EXPECTED_DENDRITIC_ORDER:
            return

        message = (
            "Scientific contract violated: expected "
            f"{[s.value for s in EXPECTED_DENDRITIC_ORDER]}, got "
            f"{[s.value for s in observed]}"
        )
        if self.level == "strict":
            raise ScientificContractViolation(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
