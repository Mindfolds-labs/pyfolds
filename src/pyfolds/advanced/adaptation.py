"""Mixin para adaptação (Spike-Frequency Adaptation - SFA)."""

import math
import torch
from typing import Dict
from ..utils.types import LearningMode


class AdaptationMixin:
    """
    Mixin para adaptação de frequência de disparo (SFA).
    
    ✅ SEMÂNTICA: POR AMOSTRA (batch independente)
        - adaptation_current é [B] (um por amostra)
        - Cada amostra tem sua própria corrente de adaptação
    
    Baseado em:
        - Benda & Herz (2007) - A universal model for spike-frequency adaptation
    """
    
    def _init_adaptation(self, cfg):
        """
        Inicializa parâmetros de adaptação a partir da config.
        
        Args:
            cfg: MPJRDConfig com parâmetros de adaptação
        """
        self.adaptation_increment = cfg.adaptation_increment
        self.adaptation_decay = cfg.adaptation_decay
        self.adaptation_max = cfg.adaptation_max
        self.adaptation_tau = cfg.adaptation_tau
        
        # ✅ Estado por batch (inicializado no forward)
        self.adaptation_current = None  # Será [B]
    
    def _ensure_adaptation_current(self, batch_size: int, device: torch.device):
        """Garante que adaptation_current existe com tamanho correto."""
        if (self.adaptation_current is None or 
            self.adaptation_current.shape[0] != batch_size):
            self.adaptation_current = torch.zeros(batch_size, device=device)
    
    def _apply_adaptation(self, u: torch.Tensor, spikes: torch.Tensor,
                           dt: float = 1.0) -> torch.Tensor:
        """
        Aplica corrente de adaptação ao potencial (POR AMOSTRA).
        
        Args:
            u: Potencial somático [B]
            spikes: Spikes do passo atual [B]
            dt: Passo de tempo (ms)
        
        Returns:
            Potencial adaptado [B]
        """
        batch_size = u.shape[0]
        device = u.device
        
        self._ensure_adaptation_current(batch_size, device)
        
        # ✅ CORRIGIDO: decaimento escalar com math.exp
        decay = math.exp(-dt / self.adaptation_tau)
        self.adaptation_current.mul_(decay)
        
        # ✅ CORRIGIDO: incrementa onde houve spike (por amostra)
        spike_mask = spikes > 0.5  # [B] booleano
        increment = self.adaptation_increment * spike_mask.float()
        self.adaptation_current.add_(increment)
        self.adaptation_current.clamp_(max=self.adaptation_max)
        
        return u - self.adaptation_current
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com adaptação POR AMOSTRA."""
        output = super().forward(x, **kwargs)
        
        if kwargs.get('mode') != LearningMode.INFERENCE:
            u_adapted = self._apply_adaptation(
                output['u'], 
                output['spikes'],
                dt=kwargs.get('dt', 1.0)
            )
            
            theta = output.get('theta', getattr(self, 'theta', torch.tensor(4.5)))
            spikes_adapted = (u_adapted >= theta).float()
            
            output['u_adapted'] = u_adapted
            output['spikes'] = spikes_adapted
            output['adaptation_current'] = self.adaptation_current.clone()
        
        return output