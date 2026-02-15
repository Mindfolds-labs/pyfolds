"""Mixin para período refratário."""

import torch
from typing import Dict, Any, Optional, Tuple
from .time_mixin import TimedMixin


class RefractoryMixin(TimedMixin):
    """
    Mixin para período refratário.
    
    ✅ SEMÂNTICA: POR AMOSTRA (batch independente)
        - Cada amostra tem seu próprio estado refratário
        - last_spike_time é [B] (um por amostra)
        - Refratário aplicado individualmente
    """
    
    def _init_refractory(self, t_refrac_abs: float = 2.0,
                          t_refrac_rel: float = 5.0,
                          refrac_rel_strength: float = 3.0):
        self.t_refrac_abs = t_refrac_abs
        self.t_refrac_rel = t_refrac_rel
        self.refrac_rel_strength = refrac_rel_strength
        self._ensure_time_counter()
        self.last_spike_time = None 

    def _ensure_last_spike_time(self, batch_size: int, device: torch.device):
        if (self.last_spike_time is None or 
            self.last_spike_time.shape[0] != batch_size):
            self.last_spike_time = torch.full(
                (batch_size,), -1000.0, device=device
            )
    
    def _check_refractory_batch(self, current_time: float, 
                                 batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time_since = current_time - self.last_spike_time

        # Refratário absoluto: nenhum spike
        in_absolute = time_since <= self.t_refrac_abs

        # Refratário relativo: threshold elevado
        in_relative = (
            (time_since > self.t_refrac_abs) &
            (time_since <= self.t_refrac_rel)
        )

        blocked = in_absolute | in_relative
        theta_boost = torch.where(
            in_relative,
            torch.full_like(time_since, self.refrac_rel_strength),
            torch.zeros_like(time_since)
        )
        return blocked, theta_boost
    
    def _update_refractory_batch(self, spikes: torch.Tensor, dt: float = 1.0):
        current_time = self.time_counter.item()
        spike_mask = spikes > 0.5
        self.last_spike_time = torch.where(
            spike_mask,
            torch.full_like(self.last_spike_time, current_time),
            self.last_spike_time
        )
        self._increment_time(dt)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        self._ensure_last_spike_time(batch_size, device)
        
        output = super().forward(x, **kwargs)
        
        current_time = self.time_counter.item()
        blocked, theta_boost = self._check_refractory_batch(current_time, batch_size)
        
        # ✅ CORRIGIDO: de 'spights' para 'spikes'
        spikes = output['spikes'].clone()
        
        spikes = torch.where(blocked, torch.zeros_like(spikes), spikes)
        theta_eff = output['theta'] + theta_boost.unsqueeze(1)
        spikes_rel = (output['u'] >= theta_eff).float()
        
        final_spikes = torch.where(blocked, torch.zeros_like(spikes), spikes_rel)
        output['spikes'] = final_spikes
        output['refrac_blocked'] = blocked
        output['theta_boost'] = theta_boost
        
        self._update_refractory_batch(final_spikes, dt=kwargs.get('dt', 1.0))
        return output
