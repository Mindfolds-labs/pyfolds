"""Mixin para backpropagação dendrítica."""

import math
import torch
from collections import deque
from typing import Dict, Any, Optional
from .time_mixin import TimedMixin


class BackpropMixin(TimedMixin):
    """
    Mixin para backpropagação dendrítica.
    
    ✅ SEMÂNTICA: MISTA
        - Amplificação dendrítica: GLOBAL (média do batch)
        - Backprop trace: POR AMOSTRA (cada sinapse tem seu traço)
    
    Baseado em:
        - Stuart & Sakmann (1994) - Active propagation of somatic action potentials
    """
    
    def _init_backprop(self, cfg):
        """
        Inicializa parâmetros de backpropagação a partir da config.
        
        Args:
            cfg: MPJRDConfig com parâmetros de backprop
        """
        self.backprop_delay = cfg.backprop_delay
        self.backprop_signal = cfg.backprop_signal
        self.backprop_amp_tau = cfg.backprop_amp_tau
        self.backprop_trace_tau = cfg.backprop_trace_tau
        self.backprop_max_amp = cfg.backprop_max_amp
        self.backprop_max_gain = cfg.backprop_max_gain
        
        self._ensure_time_counter()
        
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        
        # Amplificação dendrítica: GLOBAL (média do batch)
        self.register_buffer("dendrite_amplification", torch.zeros(D))
        
        # Backprop trace: POR AMOSTRA [B, D, S] (inicializado no forward)
        self.backprop_trace = None
        
        # Fila de backprop
        max_queue_size = max(100, int(self.backprop_delay * 50))
        self.backprop_queue = deque(maxlen=max_queue_size)
    
    def _ensure_backprop_trace(self, batch_size: int, device: torch.device):
        """Garante que backprop_trace existe com tamanho correto."""
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        
        if (self.backprop_trace is None or 
            self.backprop_trace.shape[0] != batch_size):
            self.backprop_trace = torch.zeros(batch_size, D, S, device=device)
    
    def _schedule_backprop(self, spike_time: float, v_dend: torch.Tensor):
        """Agenda evento de backpropagação."""
        self.backprop_queue.append({
            'time': spike_time,
            'v_dend': v_dend.detach().clone()
        })
    
    def _process_backprop_queue(self, current_time: float):
        """Processa eventos de backpropagação pendentes."""
        # ✅ CORRIGIDO: decaimento escalar com math.exp
        decay_amp = math.exp(-1.0 / self.backprop_amp_tau)
        decay_trace = math.exp(-1.0 / self.backprop_trace_tau)
        
        self.dendrite_amplification.mul_(decay_amp)
        
        if self.backprop_trace is not None:
            self.backprop_trace.mul_(decay_trace)
        
        while self.backprop_queue and current_time >= self.backprop_queue[0]['time']:
            event = self.backprop_queue.popleft()
            v_dend = event['v_dend']  # [B, D]
            batch_size = v_dend.shape[0]
            device = v_dend.device
            
            self._ensure_backprop_trace(batch_size, device)
            
            # Amplificação GLOBAL (média do batch)
            activity_factor = torch.sigmoid(v_dend / 5.0)  # [B, D]
            amplification_gain = self.backprop_signal * activity_factor.mean(dim=0)
            self.dendrite_amplification.add_(
                amplification_gain.clamp(max=self.backprop_max_amp)
            )
            
            # ✅ CORRIGIDO: backprop trace POR AMOSTRA
            for d_idx in range(self.cfg.n_dendrites):
                # Para cada amostra, verifica se este dendrito estava ativo
                active_samples = v_dend[:, d_idx] > 0.1  # [B]
                self.backprop_trace[active_samples, d_idx, :] += self.backprop_signal
                self.backprop_trace.clamp_(max=2.0)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com backpropagação."""
        self._increment_time(kwargs.get('dt', 1.0))
        current_time = self.time_counter.item()
        
        self._process_backprop_queue(current_time)
        
        output = super().forward(x, **kwargs)
        
        if output['spikes'].any():
            self._schedule_backprop(
                current_time + self.backprop_delay,
                output['v_dend']
            )
        
        output['dendrite_amplification'] = self.dendrite_amplification.clone()
        if self.backprop_trace is not None:
            output['backprop_trace_mean'] = self.backprop_trace.mean()
        
        return output