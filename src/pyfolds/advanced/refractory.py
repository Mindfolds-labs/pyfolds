"""Mixin para período refratário."""

import torch
from typing import Dict, Tuple
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
        """
        Inicializa parâmetros do período refratário.
        
        Args:
            t_refrac_abs: Duração do refratário absoluto (ms)
            t_refrac_rel: Duração do refratário relativo (ms)
            refrac_rel_strength: Força do boost no threshold durante refratário relativo
        """
        self.t_refrac_abs = t_refrac_abs
        self.t_refrac_rel = t_refrac_rel
        self.refrac_rel_strength = refrac_rel_strength
        self._ensure_time_counter()
        self.last_spike_time = None

    def _ensure_last_spike_time(self, batch_size: int, device: torch.device):
        """
        Garante que last_spike_time existe com tamanho correto.
        
        Args:
            batch_size: Tamanho do batch
            device: Device onde o tensor deve ser alocado
        """
        if (self.last_spike_time is None or 
            self.last_spike_time.shape[0] != batch_size):
            self.last_spike_time = torch.full(
                (batch_size,), -1000.0, device=device
            )
    
    def _check_refractory_batch(self, current_time: float, 
                                 batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verifica estado refratário para todas as amostras do batch.
        
        Args:
            current_time: Tempo atual
            batch_size: Tamanho do batch
            
        Returns:
            Tuple com:
                - blocked: [B] máscara de bloqueio (absoluto + relativo)
                - theta_boost: [B] boost no threshold (apenas relativo)
        """
        time_since = current_time - self.last_spike_time
        
        # Refratário absoluto: bloqueia spikes imediatamente após disparo
        in_absolute = time_since <= self.t_refrac_abs

        # Refratário relativo: não bloqueia diretamente; aumenta threshold
        in_relative = (
            (time_since > self.t_refrac_abs) &
            (time_since <= self.t_refrac_rel)
        )

        blocked = in_absolute

        # Boost no threshold durante refratário relativo
        theta_boost = torch.where(
            in_relative,
            torch.full_like(time_since, self.refrac_rel_strength),
            torch.zeros_like(time_since),
        )

        return blocked, theta_boost
    
    def _update_refractory_batch(self, spikes: torch.Tensor, dt: float = 1.0):
        """
        Atualiza estado refratário baseado nos spikes.
        
        Args:
            spikes: [B] spikes do passo atual
            dt: Passo de tempo (ms)
        """
        current_time = self.time_counter.item()
        spike_mask = spikes > 0.5
        
        # Atualiza último spike time onde houve disparo
        self.last_spike_time = torch.where(
            spike_mask,
            torch.full_like(self.last_spike_time, current_time),
            self.last_spike_time
        )
        
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass com período refratário.
        
        Args:
            x: Tensor de entrada [B, D, S]
            **kwargs: Argumentos adicionais (dt, etc.)
            
        Returns:
            Dict com spikes após aplicação do refratário
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Garante que last_spike_time existe
        self._ensure_last_spike_time(batch_size, device)
        
        # Forward da classe base
        output = super().forward(x, **kwargs)
        
        # Verifica estado refratário
        current_time = self.time_counter.item()
        blocked, theta_boost = self._check_refractory_batch(current_time, batch_size)
        
        # Aplica refratário
        # theta_eff deve permanecer 1-D ([B]) para preservar a semântica
        # ponto-a-ponto do refratário por amostra.
        theta_raw = output['theta']
        if theta_raw.dim() == 0:
            theta = torch.full_like(theta_boost, theta_raw.item())
        elif theta_raw.dim() == 1 and theta_raw.shape[0] == 1:
            theta = theta_raw.expand(batch_size)
        elif theta_raw.dim() == 1 and theta_raw.shape[0] == batch_size:
            theta = theta_raw
        else:
            raise ValueError(
                "Campo 'theta' incompatível para refratário: "
                f"shape={tuple(theta_raw.shape)}, esperado escalar, [1] ou [{batch_size}]"
            )

        theta_eff = theta + theta_boost
        spikes_rel = (output['u'] >= theta_eff).float()
        
        # Bloqueia spikes apenas no refratário absoluto
        final_spikes = torch.where(blocked, torch.zeros_like(spikes_rel), spikes_rel)
        
        # Atualiza output
        output['spikes'] = final_spikes
        output['refrac_blocked'] = blocked
        output['theta_boost'] = theta_boost
        
        # Atualiza estado refratário
        self._update_refractory_batch(final_spikes, dt=kwargs.get('dt', 1.0))
        
        return output
    
    def reset_refractory(self):
        """Reseta o estado refratário."""
        self.last_spike_time = None
        # Nota: time_counter NÃO é resetado aqui pois é compartilhado
    
    def get_refractory_metrics(self) -> dict:
        """
        Retorna métricas do período refratário.
        
        Returns:
            dict: Dicionário com métricas
        """
        metrics = {
            't_refrac_abs': self.t_refrac_abs,
            't_refrac_rel': self.t_refrac_rel,
            'refrac_rel_strength': self.refrac_rel_strength,
        }
        
        if self.last_spike_time is not None:
            metrics['last_spike_time_mean'] = self.last_spike_time.mean().item()
            metrics['last_spike_time_min'] = self.last_spike_time.min().item()
            metrics['last_spike_time_max'] = self.last_spike_time.max().item()
        
        return metrics
