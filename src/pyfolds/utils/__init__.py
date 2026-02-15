"""Utilitários para PyFolds

Este módulo fornece funções auxiliares e tipos para todo o framework:
- Funções matemáticas seguras (divisão, clamps, inicialização)
- Gerenciamento de device (CPU/GPU)
- Modos de aprendizado e configurações
- Logging profissional (TRACE, DEBUG, INFO, WARNING)

Uso básico:
    from pyfolds.utils import safe_div, clamp_rate, get_device, LearningMode, get_logger
    
    device = get_device()
    x = torch.randn(10).to(device)
    y = torch.randn(10).to(device)
    z = safe_div(x, y)
    
    mode = LearningMode.ONLINE
    print(mode.description)
    
    # Logging
    logger = get_logger(__name__)
    logger.info("Mensagem informativa")
    logger.trace("Mensagem de trace")  # Nível personalizado
"""

from .math import safe_div, clamp_rate, clamp_R, xavier_init, calculate_vc_dimension
from .device import infer_device, ensure_device, get_device
from .types import LearningMode, ConnectionType, ModeConfig

# ✅ ADICIONADO: Logging
from .logging import get_logger, PyFoldsLogger, trace, TRACE_LEVEL

__all__ = [
    # Math
    "safe_div",
    "clamp_rate", 
    "clamp_R",
    "xavier_init",
    "calculate_vc_dimension",
    
    # Device
    "infer_device",
    "ensure_device", 
    "get_device",
    
    # Types
    "LearningMode",
    "ConnectionType",
    "ModeConfig",
    
    # ✅ ADICIONADO: Logging
    "get_logger",
    "PyFoldsLogger",
    "trace",
    "TRACE_LEVEL",
]