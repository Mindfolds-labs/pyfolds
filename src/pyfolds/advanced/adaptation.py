"""Mixin para adaptação (Spike-Frequency Adaptation - SFA)."""

import math
import torch
from typing import Dict
from ..utils.types import LearningMode, normalize_learning_mode


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
    
    def _apply_sfa_before_threshold(self, u: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Aplica SFA antes do threshold (u_eff = u - I_adapt).
        
        Args:
            u: Potencial somático [B]
            dt: Passo de tempo (ms)
        
        Returns:
            Potencial adaptado [B]
        """
        batch_size = u.shape[0]
        device = u.device
        
        self._ensure_adaptation_current(batch_size, device)
        
        # Decaimento ocorre antes da comparação com threshold.
        decay = math.exp(-dt / self.adaptation_tau)
        self.adaptation_current.mul_(decay)

        return u - self.adaptation_current

    def _update_adaptation_after_spike(self, spikes: torch.Tensor) -> None:
        """Atualiza I_adapt após spikes confirmados (pós-refratário)."""
        spike_mask = spikes > 0.5
        increment = self.adaptation_increment * spike_mask.float()
        self.adaptation_current.add_(increment)
        self.adaptation_current.clamp_(max=self.adaptation_max)

    # Compatibilidade retroativa para testes/código legado.
    def _apply_adaptation(self, u: torch.Tensor, spikes: torch.Tensor,
                           dt: float = 1.0) -> torch.Tensor:
        """Compat: aplica SFA antes do threshold e atualiza com spikes."""
        u_eff = self._apply_sfa_before_threshold(u, dt=dt)
        self._update_adaptation_after_spike(spikes)
        return u_eff
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com adaptação POR AMOSTRA."""
        output = super().forward(x, **kwargs)

        mode_val = normalize_learning_mode(kwargs.get('mode'))
        if mode_val != LearningMode.INFERENCE and self.adaptation_current is not None:
            output['adaptation_current'] = self.adaptation_current.clone()

        return output
