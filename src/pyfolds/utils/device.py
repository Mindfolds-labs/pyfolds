"""Gerenciamento de device para PyFolds - VERSÃO CORRIGIDA."""

import torch
import logging
from typing import Dict, Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)


def infer_device(
    inputs: Union[Dict[str, torch.Tensor], torch.Tensor, None] = None
) -> torch.device:
    """Infere device a partir de inputs."""
    if inputs is None:
        return torch.device('cpu')
    
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    
    if isinstance(inputs, dict):
        for tensor in inputs.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.device
    
    return torch.device('cpu')


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Retorna melhor device disponível."""
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Garante que tensor está no device especificado."""
    if device is None:
        return tensor
    return tensor.to(device)


class DeviceManager:
    """Gerenciador de device para PyFolds."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self._validate_device()
        logger.info(f"DeviceManager inicializado: {self.device}")
    
    def _validate_device(self) -> None:
        """Valida que device está disponível."""
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA não disponível (device solicitado: {self.device})")
    
    def to(self, *objs: Any) -> Union[Any, Tuple[Any, ...]]:
        """
        Move objetos para o device gerenciado.
        
        Args:
            *objs: Objetos a mover (tensores, módulos, etc)
        
        Returns:
            Objeto único ou tupla de objetos movidos
        """
        if not objs:
            return None
        
        moved = []
        for obj in objs:
            if hasattr(obj, 'to'):
                moved.append(obj.to(self.device))
            else:
                moved.append(obj)
        
        if len(moved) == 1:
            return moved[0]
        return tuple(moved)
    
    def check_consistency(self, *tensors: torch.Tensor) -> bool:
        """
        Verifica que todos os tensores estão no device correto.
        
        Args:
            *tensors: Tensores a verificar
        
        Returns:
            True se consistentes
        
        Raises:
            ValueError: Se devices não correspondem
        """
        devices = {t.device for t in tensors if isinstance(t, torch.Tensor)}
        
        if len(devices) > 1:
            raise ValueError(f"Múltiplos devices encontrados: {devices}")
        
        if devices and next(iter(devices)) != self.device:
            raise ValueError(
                f"Device mismatch: tensores em {next(iter(devices))}, "
                f"gerenciador espera {self.device}"
            )
        
        return True