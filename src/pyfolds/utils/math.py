"""Utilitários matemáticos para o PyFolds"""

import math
import torch
from typing import Union, Tuple


def safe_div(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Divisão segura com epsilon para evitar divisão por zero.
    
    Args:
        x: Numerador
        y: Denominador
        eps: Epsilon para evitar divisão por zero
    
    Returns:
        x / (y + eps)
    """
    return x / (y + eps)


def clamp_rate(r: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    """
    Mantém taxa de disparo em [0, 1].
    
    Args:
        r: Taxa de disparo (tensor ou float)
    
    Returns:
        Valor clampado em [0, 1]
    """
    if isinstance(r, torch.Tensor):
        return r.clamp(0.0, 1.0)
    return max(0.0, min(1.0, r))


def clamp_R(r: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    """
    Mantém sinal neuromodulador em [-1, 1].
    
    Args:
        r: Sinal neuromodulador (tensor ou float)
    
    Returns:
        Valor clampado em [-1, 1]
    """
    if isinstance(r, torch.Tensor):
        return r.clamp(-1.0, 1.0)
    return max(-1.0, min(1.0, r))


def xavier_init(shape: Tuple[int, ...], gain: float = 1.0) -> torch.Tensor:
    """
    Inicialização Xavier para pesos.
    
    Args:
        shape: Dimensões do tensor
        gain: Ganho (1.0 para sigmoid/tanh, sqrt(2) para ReLU)
    
    Returns:
        Tensor inicializado com distribuição normal * std
    """
    if len(shape) >= 2:
        fan_in = shape[-2]
        fan_out = shape[-1]
    else:
        fan_in = fan_out = shape[0]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return torch.randn(shape) * std


def calculate_vc_dimension(
    n_neurons: int, 
    n_dendrites: int,
    n_synapses: int, 
    avg_connections: float = 1.0
) -> float:
    """
    Calcula VC-dimension aproximado de uma rede MPJRD.
    
    A VC-dimension mede a capacidade de aprendizado da rede.
    Fórmula: VC-dim ≈ N * D * log2(N * D * S * C)
    
    Args:
        n_neurons: Número de neurônios
        n_dendrites: Número de dendritos por neurônio
        n_synapses: Número de sinapses por dendrito
        avg_connections: Média de conexões ativas (default: 1.0)
    
    Returns:
        VC-dimension aproximada
    
    Example:
        >>> vc = calculate_vc_dimension(100, 4, 32)
        >>> print(f"VC-dimension: {vc:.0f}")
    """
    total_params = n_neurons * n_dendrites * n_synapses * avg_connections
    return n_neurons * n_dendrites * math.log2(max(1, total_params))