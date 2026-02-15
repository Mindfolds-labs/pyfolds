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
            ONLINE: 5.0 (aprendizado acelerado)
            BATCH: 0.2 (atualização conservadora)
            SLEEP: 0.0 (sem plasticidade ativa)
            INFERENCE: 0.0 (sem aprendizado)
        """
        multipliers = {
            LearningMode.ONLINE: 5.0,
            LearningMode.BATCH: 0.2,
            LearningMode.SLEEP: 0.0,
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
    ENTANGLED = "entangled"  # Acoplamento recorrente/híbrido


# ============================================================================
# CONFIGURAÇÕES - ✅ ModeConfig ADICIONADO
# ============================================================================

@dataclass
class ModeConfig:
    """
    Configurações específicas por modo de aprendizado.
    
    Configuração global para escalonar parâmetros por `LearningMode`.
    """
    online_learning_rate_mult: float = 5.0
    batch_learning_rate_mult: float = 0.2
    sleep_consolidation_factor: float = 0.1
    
    def __post_init__(self):
        """Valida configuração."""
        if self.online_learning_rate_mult < 0.0:
            raise ValueError("online_learning_rate_mult deve ser >= 0")
        if self.batch_learning_rate_mult < 0.0:
            raise ValueError("batch_learning_rate_mult deve ser >= 0")
        if not 0.0 <= self.sleep_consolidation_factor <= 1.0:
            raise ValueError("sleep_consolidation_factor deve estar em [0, 1]")

    def get_learning_rate(self, base_lr: float, mode: LearningMode) -> float:
        """Retorna learning rate efetivo para o modo."""
        if mode == LearningMode.ONLINE:
            return base_lr * self.online_learning_rate_mult
        if mode == LearningMode.BATCH:
            return base_lr * self.batch_learning_rate_mult
        return 0.0

    def get_consolidation_factor(self, mode: LearningMode) -> float:
        """Retorna fator de consolidação no modo sono."""
        if mode == LearningMode.SLEEP:
            return self.sleep_consolidation_factor
        return 0.0
    
    @classmethod
    def from_learning_mode(cls, mode: LearningMode) -> 'ModeConfig':
        """Cria config com foco no modo informado (compatibilidade)."""
        cfg = cls()
        if mode == LearningMode.ONLINE:
            cfg.batch_learning_rate_mult = 0.0
        elif mode == LearningMode.BATCH:
            cfg.online_learning_rate_mult = 0.0
        elif mode in (LearningMode.SLEEP, LearningMode.INFERENCE):
            cfg.online_learning_rate_mult = 0.0
            cfg.batch_learning_rate_mult = 0.0
        return cfg


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
