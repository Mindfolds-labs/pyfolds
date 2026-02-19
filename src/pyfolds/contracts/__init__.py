"""Contrato canônico de execução de passo do neurônio."""

from .neuron_contract import (
    CONTRACT_MECHANISM_ORDER,
    MechanismStep,
    NeuronStepInput,
    NeuronStepOutput,
    StepExecutionTrace,
    validate_step_output,
)
from .backends import TorchNeuronContractBackend, TensorFlowNeuronContractBackend

__all__ = [
    "CONTRACT_MECHANISM_ORDER",
    "MechanismStep",
    "NeuronStepInput",
    "NeuronStepOutput",
    "StepExecutionTrace",
    "validate_step_output",
    "TorchNeuronContractBackend",
    "TensorFlowNeuronContractBackend",
]
