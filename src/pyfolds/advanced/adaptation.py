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
    
    def _adaptation_load_state_dict_pre_hook(self, module, state_dict, prefix, *args):
        key = f"{prefix}adaptation_current"
        if key in state_dict:
            self.adaptation_current.resize_(state_dict[key].shape)

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
        
        # ✅ Estado por batch (buffer persistente; inicializado no forward)
        if not hasattr(self, "adaptation_current"):
            self.register_buffer("adaptation_current", torch.empty(0))
        if not getattr(self, "_adaptation_state_hook_registered", False):
            self.register_load_state_dict_pre_hook(self._adaptation_load_state_dict_pre_hook)
            self._adaptation_state_hook_registered = True
    
    def _ensure_adaptation_current(self, batch_size: int, device: torch.device):
        """Garante que adaptation_current existe com tamanho correto."""
        needs_resize = self.adaptation_current.ndim != 1 or self.adaptation_current.shape[0] != batch_size
        if needs_resize:
            self.adaptation_current.resize_(batch_size)
            self.adaptation_current.zero_()
    
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

        self._ensure_adaptation_current(batch_size, u.device)
        
        # Decaimento ocorre antes da comparação com threshold.
        decay = math.exp(-dt / self.adaptation_tau)
        self.adaptation_current.mul_(decay)

        return u - self.adaptation_current

    def _update_adaptation_after_spike(self, spikes: torch.Tensor) -> None:
        """Atualiza I_adapt após spikes confirmados (pós-refratário).

        Parameters
        ----------
        spikes : torch.Tensor
            Tensor de spikes por amostra.
        """
        if self.adaptation_current.numel() == 0:
            self._ensure_adaptation_current(int(spikes.shape[0]), spikes.device)

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

        # SFA é aplicada no ponto de decisão de spike (RefractoryMixin),
        # evitando dupla aplicação e ordem ambígua no pós-forward.
        mode_val = normalize_learning_mode(kwargs.get('mode'))
        if mode_val != LearningMode.INFERENCE and self.adaptation_current.numel() > 0:
            output['adaptation_current'] = self.adaptation_current.clone()

        return output
