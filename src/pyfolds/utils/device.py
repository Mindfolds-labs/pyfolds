"""Gerenciamento de device para PyFolds"""

import torch
from typing import Dict, Optional, Union


def infer_device(
    inputs: Union[Dict[str, torch.Tensor], torch.Tensor, None] = None
) -> torch.device:
    """
    Infere o device a partir de inputs ou retorna CPU como fallback.
    
    Args:
        inputs: Tensor, dicionário de tensores ou None
    
    Returns:
        torch.device: Device inferido
    
    Example:
        >>> x = torch.randn(10, device='cuda')
        >>> device = infer_device(x)  # returns device(type='cuda')
        >>> device = infer_device()    # returns device(type='cpu')
    """
    if inputs is None:
        return torch.device('cpu')
    
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    
    if isinstance(inputs, dict):
        for tensor in inputs.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
    
    return torch.device('cpu')


def ensure_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Garante que o tensor está no device especificado.
    
    Args:
        tensor: Tensor a ser movido
        device: Device destino (se None, retorna original)
    
    Returns:
        Tensor no device correto
    
    Example:
        >>> x = torch.randn(10)
        >>> x = ensure_device(x, torch.device('cuda'))
    """
    if device is None:
        return tensor
    return tensor.to(device)


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Retorna o melhor device disponível (CUDA se possível, CPU caso contrário).
    
    Args:
        device: Device específico (opcional)
    
    Returns:
        torch.device: Device configurado
    
    Example:
        >>> device = get_device()  # 'cuda' se disponível, senão 'cpu'
        >>> device = get_device('cpu')  # Força CPU
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')