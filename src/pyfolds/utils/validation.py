"""Utilitários para validação de entradas de modelos."""

from __future__ import annotations

from functools import wraps
from typing import Callable, Optional, Tuple

import torch


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

