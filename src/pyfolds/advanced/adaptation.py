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
        """Aplica SFA antes do threshold, sem recalcular spikes."""
        if dt <= 0:
            raise ValueError(f"dt deve ser positivo para adaptação, recebido {dt}")

        batch_size = u.shape[0]
        device = u.device

        self._ensure_adaptation_current(batch_size, device)

        decay = math.exp(-dt / self.adaptation_tau)
        self.adaptation_current.mul_(decay)

        return u - self.adaptation_current

    def _update_adaptation_from_spikes(self, spikes: torch.Tensor) -> None:
        """Atualiza corrente de adaptação com spikes finais confirmados."""
        spike_mask = spikes > 0.5  # [B] booleano
        increment = self.adaptation_increment * spike_mask.float()
        self.adaptation_current.add_(increment)
        self.adaptation_current.clamp_(max=self.adaptation_max)

    def _apply_adaptation(self, u: torch.Tensor, spikes: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Compat: aplica SFA pré-threshold e depois atualiza corrente por spikes."""
        u_eff = self._apply_sfa_before_threshold(u, dt=dt)
        self._update_adaptation_from_spikes(spikes)
        return u_eff
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com adaptação POR AMOSTRA."""
        output = super().forward(x, **kwargs)
        
        mode_val = normalize_learning_mode(kwargs.get('mode'))
        if mode_val != LearningMode.INFERENCE:
            required_fields = ('u', 'spikes')
            missing = [field for field in required_fields if field not in output]
            if missing:
                raise KeyError(
                    "Campos obrigatórios ausentes em output para adaptação: "
                    f"{missing}. Campos disponíveis: {list(output.keys())}"
                )
            self._ensure_adaptation_current(output['u'].shape[0], output['u'].device)
            self._update_adaptation_from_spikes(output['spikes'])

            if 'u_raw' not in output:
                output['u_raw'] = output['u']
            if 'u_adapted' not in output:
                output['u_adapted'] = output['u']
            output['adaptation_current'] = self.adaptation_current.clone()
        
        return output
