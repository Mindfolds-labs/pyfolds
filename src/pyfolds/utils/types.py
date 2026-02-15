"""Tipos de configuração para PyFolds - VERSÃO COMPLETA."""

from dataclasses import dataclass
from typing import NamedTuple, Dict, Optional
from enum import Enum
import torch
from torch import Tensor

# ============================================================================
# ENUMS PRINCIPAIS - ✅ ADICIONADOS
# ============================================================================

class LearningMode(Enum):
    """Modos de aprendizado do neurônio MPJRD."""
    
    ONLINE = "online"
    BATCH = "batch"
    SLEEP = "sleep"
    INFERENCE = "inference"
    
    @property
    def learning_rate_multiplier(self) -> float:
        """
        Multiplicador de learning rate para cada modo.
        
        Returns:
            ONLINE: 1.0 (taxa normal)
            BATCH: 0.8 (80% da taxa)
            SLEEP: 0.1 (10% - consolidação)
            INFERENCE: 0.0 (sem aprendizado)
        """
        multipliers = {
            LearningMode.ONLINE: 1.0,
            LearningMode.BATCH: 0.8,
            LearningMode.SLEEP: 0.1,
            LearningMode.INFERENCE: 0.0,
        }
        return multipliers[self]
    
    @property
    def description(self) -> str:
        """Descrição do modo de aprendizado."""
        descriptions = {
            LearningMode.ONLINE: "Atualização imediata (vigília)",
            LearningMode.BATCH: "Acumula e atualiza em lote (consolidação)",
            LearningMode.SLEEP: "Sono - two-factor consolidation (I → N)",
            LearningMode.INFERENCE: "Modo produção - sem aprendizado",
        }
        return descriptions[self]
    
    def is_learning(self) -> bool:
        """Retorna True se o modo permite aprendizado."""
        return self in [LearningMode.ONLINE, LearningMode.BATCH]
    
    def is_consolidating(self) -> bool:
        """Retorna True se o modo é consolidação (sono)."""
        return self == LearningMode.SLEEP


class ConnectionType(Enum):
    """Tipos de conexão entre camadas."""
    
    DENSE = "dense"          # Totalmente conectado
    SPARSE = "sparse"        # Conectado esparsamente
    EXCITATORY = "exc"       # Excitatório
    INHIBITORY = "inh"       # Inibitório
    MODULATORY = "mod"       # Modulatório (neuromodulador)


# ============================================================================
# CONFIGURAÇÕES - ✅ ModeConfig ADICIONADO
# ============================================================================

@dataclass
class ModeConfig:
    """
    Configurações específicas por modo de aprendizado.
    
    Attributes:
        name: Nome do modo
        learning_rate_multiplier: Multiplicador para learning rate
        description: Descrição do modo
    """
    name: str
    learning_rate_multiplier: float = 1.0
    description: str = ""
    
    def __post_init__(self):
        """Valida configuração."""
        if not 0.0 <= self.learning_rate_multiplier <= 2.0:
            raise ValueError(
                f"learning_rate_multiplier deve estar em [0, 2], "
                f"got {self.learning_rate_multiplier}"
            )
    
    @classmethod
    def from_learning_mode(cls, mode: LearningMode) -> 'ModeConfig':
        """Cria ModeConfig a partir de LearningMode."""
        return cls(
            name=mode.value,
            learning_rate_multiplier=mode.learning_rate_multiplier,
            description=mode.description
        )


# ============================================================================
# TYPE ALIASES
# ============================================================================

TensorBatch = Tensor  # [B, ...]
TensorShared = Tensor  # [D, S]
DeviceType = torch.device


# ============================================================================
# ADAPTAÇÃO
# ============================================================================

class AdaptationOutput(NamedTuple):
    """Output de _apply_adaptation."""
    u_adapted: TensorBatch  # [B]
    adaptation_current: TensorBatch  # [B]
    metrics: Dict[str, float]


@dataclass
class AdaptationConfig:
    """Configuração para adaptação de neurônios."""
    
    adaptation_increment: float = 0.1
    """Incremento de corrente de adaptação por spike [nA]."""
    
    adaptation_decay: float = 0.98
    """Fator de decaimento (0.0-1.0)."""
    
    adaptation_max: float = 1.0
    """Máximo valor de corrente de adaptação [nA]."""
    
    adaptation_tau: float = 100.0
    """Constante de tempo de decaimento [ms]."""
    
    def validate(self) -> None:
        """Valida parâmetros."""
        errors = []
        
        if not 0.0 < self.adaptation_increment <= 1.0:
            errors.append(f"adaptation_increment={self.adaptation_increment} "
                         f"must be in (0.0, 1.0]")
        
        if not 0.0 <= self.adaptation_decay <= 1.0:
            errors.append(f"adaptation_decay={self.adaptation_decay} "
                         f"must be in [0.0, 1.0]")
        
        if self.adaptation_max <= 0:
            errors.append(f"adaptation_max must be > 0, got {self.adaptation_max}")
        
        if self.adaptation_tau <= 0:
            errors.append(f"adaptation_tau must be > 0, got {self.adaptation_tau}")
        
        if errors:
            raise ValueError("\n".join(errors))