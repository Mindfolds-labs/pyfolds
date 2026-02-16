"""Utilitários para validação de entradas de modelos."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Tuple

import torch


def validate_device_consistency(*tensors: torch.Tensor) -> torch.device:
    """Valida se todos os tensores informados estão no mesmo device.

    Args:
        *tensors: Tensors opcionais a serem validados.

    Returns:
        O ``torch.device`` compartilhado pelos tensores válidos.

    Raises:
        ValueError: Se nenhum tensor for informado.
        RuntimeError: Se houver inconsistência de devices.
    """

    valid_tensors = [t for t in tensors if t is not None]
    if not valid_tensors:
        raise ValueError("É necessário informar ao menos um tensor para validação de device")

    devices = {tensor.device for tensor in valid_tensors}
    if len(devices) > 1:
        raise RuntimeError(f"Tensores em devices diferentes: {devices}")

    return valid_tensors[0].device


def validate_input(
    *,
    expected_ndim: Optional[int] = None,
    expected_shape_fn: Optional[Callable[[object], Tuple[int, int]]] = None,
    value_range: Optional[Tuple[float, float]] = None,
):
    """Decorator para validação de tensores de entrada.

    Args:
        expected_ndim: Número esperado de dimensões.
        expected_shape_fn: Função que recebe ``self`` e retorna shape esperado
            para os eixos [dendrites, synapses].
        value_range: Faixa permitida para os valores do tensor.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, x, *args, **kwargs):
            if not isinstance(x, torch.Tensor):
                raise TypeError("x deve ser um torch.Tensor")

            if expected_ndim is not None and x.ndim != expected_ndim:
                raise ValueError(f"x deve ter {expected_ndim} dimensões, recebido {x.ndim}")

            if expected_shape_fn is not None:
                exp_d, exp_s = expected_shape_fn(self)
                if x.shape[1:] != (exp_d, exp_s):
                    raise ValueError(
                        f"Esperado ({exp_d}, {exp_s}) em x.shape[1:], recebido {tuple(x.shape[1:])}"
                    )

            if not torch.is_floating_point(x):
                raise TypeError("x deve ser tensor de ponto flutuante")

            if value_range is not None:
                min_val, max_val = value_range
                x_min = float(x.min().item())
                x_max = float(x.max().item())
                if x_min < min_val or x_max > max_val:
                    raise ValueError(f"Valores de x fora da faixa [{min_val}, {max_val}]")

            return func(self, x, *args, **kwargs)

        return wrapper

    return decorator


def validate_device_consistency(*tensors: torch.Tensor) -> torch.device:
    """Valida se todos os tensores informados estão no mesmo device.

    Args:
        *tensors: Tensores a serem verificados.

    Returns:
        Device comum encontrado.

    Raises:
        ValueError: Se nenhum tensor for informado.
        TypeError: Se algum item não for ``torch.Tensor``.
        RuntimeError: Se existir inconsistência entre devices.
    """
    if not tensors:
        raise ValueError("Pelo menos um tensor deve ser informado")

    ref_device: Optional[torch.device] = None
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Todos os argumentos devem ser torch.Tensor")

        if ref_device is None:
            ref_device = tensor.device
            continue

        if tensor.device != ref_device:
            raise RuntimeError(
                f"Inconsistência de devices detectada: esperado {ref_device}, recebido {tensor.device}"
            )

    return ref_device
