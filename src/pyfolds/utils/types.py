"""Tipos e enums para o PyFolds"""

from enum import Enum
from dataclasses import dataclass


class LearningMode(str, Enum):
    """
    Modos de aprendizado do neurônio MPJRD.
    
    Baseado na literatura:
    - Xiao et al., 2024 - Dual-process theory
    - Iatropoulos et al., PNAS 2025 - Two-factor consolidation
    
    Modos:
        ONLINE: Aprendizado imediato (SGD-like)
        BATCH: Aprendizado em lote (gradient accumulation)
        SLEEP: Consolidação two-factor (transferência I→N)
        INFERENCE: Apenas forward (produção)
    """
    ONLINE = "online"
    BATCH = "batch"
    SLEEP = "sleep"
    INFERENCE = "inference"
    
    @property
    def description(self) -> str:
        """Descrição do modo de aprendizado."""
        return {
            "online": "Aprendizado rápido e imediato (vigília)",
            "batch": "Aprendizado estável em lote (consolidação)",
            "sleep": "Sono - two-factor consolidation (I → N)",
            "inference": "Modo produção - sem aprendizado"
        }[self.value]
    
    @property
    def learning_rate_multiplier(self) -> float:
        """
        Multiplicador de learning rate para cada modo.
        
        Returns:
            ONLINE: 5.0 (aprende 5x mais rápido)
            BATCH: 0.2 (aprende 5x mais devagar)
            SLEEP: 0.0 (não aprende, consolida)
            INFERENCE: 0.0 (não aprende)
        """
        return {
            "online": 5.0,
            "batch": 0.2,
            "sleep": 0.0,
            "inference": 0.0
        }[self.value]
    
    def is_learning(self) -> bool:
        """Retorna True se o modo permite aprendizado."""
        return self in [LearningMode.ONLINE, LearningMode.BATCH]
    
    def is_consolidating(self) -> bool:
        """Retorna True se o modo é consolidação (sono)."""
        return self == LearningMode.SLEEP


class ConnectionType(str, Enum):
    """Tipos de conexão entre camadas."""
    DENSE = "dense"        # Totalmente conectado
    SPARSE = "sparse"      # Conectado esparsamente
    ENTANGLED = "entangled"  # Conexões emaranhadas (experimental)


@dataclass
class ModeConfig:
    """
    Configurações específicas por modo de aprendizado.
    
    Attributes:
        online_learning_rate_mult: Multiplicador para modo ONLINE
        batch_learning_rate_mult: Multiplicador para modo BATCH
        sleep_consolidation_factor: Fator de consolidação no sono
    """
    online_learning_rate_mult: float = 5.0
    batch_learning_rate_mult: float = 0.2
    sleep_consolidation_factor: float = 0.1
    
    def get_learning_rate(self, base_lr: float, mode: LearningMode) -> float:
        """
        Retorna learning rate ajustado para o modo.
        
        Args:
            base_lr: Learning rate base
            mode: Modo de aprendizado
        
        Returns:
            Learning rate ajustado (base_lr * multiplicador)
        """
        mult = {
            LearningMode.ONLINE: self.online_learning_rate_mult,
            LearningMode.BATCH: self.batch_learning_rate_mult,
            LearningMode.SLEEP: 0.0,
            LearningMode.INFERENCE: 0.0
        }[mode]
        return base_lr * mult
    
    def get_consolidation_factor(self, mode: LearningMode) -> float:
        """
        Retorna fator de consolidação para o modo.
        
        Args:
            mode: Modo de aprendizado
        
        Returns:
            Fator de consolidação (apenas SLEEP > 0)
        """
        return self.sleep_consolidation_factor if mode == LearningMode.SLEEP else 0.0