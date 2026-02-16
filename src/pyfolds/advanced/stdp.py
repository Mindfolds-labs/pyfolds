"""Mixin para STDP (Spike-Timing Dependent Plasticity)."""

import math
import torch
from typing import Dict, Any, Optional
from ..utils.types import LearningMode


class STDPMixin:
    """
    Mixin para STDP (Spike-Timing Dependent Plasticity).
    
    ✅ VETORIZADO: sem loops Python
    ✅ SEMÂNTICA: POR AMOSTRA (batch independente)
    
    Implementa plasticidade baseada em temporização de spikes:
        - LTP: spike pré antes do pós
        - LTD: spike pós antes do pré
    
    Baseado em:
        - Bi & Poo (1998) - Synaptic modifications in cultured hippocampal neurons
    """
    
    def _init_stdp(self, tau_pre: float = 20.0, tau_post: float = 20.0,
                    A_plus: float = 0.01, A_minus: float = 0.012,
                    plasticity_mode: str = "both"):
        """
        Inicializa parâmetros STDP.
        
        Args:
            tau_pre: Constante de tempo do traço pré-sináptico
            tau_post: Constante de tempo do traço pós-sináptico
            A_plus: Amplitude LTP
            A_minus: Amplitude LTD
            plasticity_mode: 'stdp', 'hebbian', 'both', 'none'
        """
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.plasticity_mode = plasticity_mode
        
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        
        # Traços POR AMOSTRA
        self.trace_pre = None  # [B, D, S]
        self.trace_post = None  # [B, D, S]
    
    def _ensure_traces(self, batch_size: int, device: torch.device):
        """Garante que os traços existem com tamanho correto."""
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite
        
        if (self.trace_pre is None or self.trace_pre.shape[0] != batch_size):
            self.trace_pre = torch.zeros(batch_size, D, S, device=device)
            self.trace_post = torch.zeros(batch_size, D, S, device=device)
    
    def _update_stdp_traces(self, x: torch.Tensor, post_spike: torch.Tensor,
                              dt: float = 1.0):
        """
        Atualiza traços STDP de forma VETORIZADA.
        
        ✅ VETORIZADO: sem loops Python
        ✅ OTIMIZADO: fallback de loop removido
        
        Args:
            x: Tensor de entrada [B, D, S]
            post_spike: Spike pós-sináptico [B]
            dt: Passo de tempo
        """
        batch_size = x.shape[0]
        device = x.device
        D, S = self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite
        
        self._ensure_traces(batch_size, device)
        
        # Decaimento escalar
        decay_pre = math.exp(-dt / self.tau_pre)
        decay_post = math.exp(-dt / self.tau_post)
        
        # Decaimento (vetorizado)
        self.trace_pre.mul_(decay_pre)
        self.trace_post.mul_(decay_post)
        
        # Spikes pré (detectados por amostra)
        pre_spikes = (x > 0.5).float()  # [B, D, S]
        
        # Adiciona aos traços pré
        self.trace_pre.add_(pre_spikes)
        
        # Spike pós (broadcast para [B, 1, 1])
        post_expanded = post_spike.view(-1, 1, 1)  # [B, 1, 1]
        
        # LTD: onde trace_post > threshold
        # Nota técnica:
        #   Nesta formulação, LTP e LTD são ambos modulados por post_expanded
        #   no passo atual. A interpretação é "ajuste condicionado ao spike
        #   pós-sináptico", e não uma implementação canônica por Δt explícito
        #   (pre-before-post vs post-before-pre). Para equivalência estrita com
        #   STDP pair-based clássico, recomenda-se validar com curvas Δw(Δt).
        ltd_mask = (self.trace_post > 0.01).float()
        delta_ltd = -self.A_minus * self.trace_post * ltd_mask * post_expanded
        
        # LTP: onde trace_pre > threshold
        ltp_mask = (self.trace_pre > 0.01).float()
        delta_ltp = self.A_plus * self.trace_pre * ltp_mask * post_expanded
        
        # ✅ VETORIZADO: aplica a TODAS as sinapses de uma vez
        # Requer que self.I exista (tensor consolidado)
        if hasattr(self, 'I'):
            # I é [D, S] - expande para batch
            I_expanded = self.I.unsqueeze(0).expand(batch_size, -1, -1)  # [B, D, S]
            I_updated = I_expanded + delta_ltd + delta_ltp
            I_updated = I_updated.clamp(self.cfg.i_min, self.cfg.i_max)
            
            # Atualiza I (média sobre batch para manter [D, S])
            self.I.copy_(I_updated.mean(dim=0))
        
        # Adiciona traço pós
        self.trace_post.add_(post_expanded)
    
    def _should_apply_stdp(self, mode: Optional[LearningMode] = None) -> bool:
        """Determina se STDP deve ser aplicado."""
        stdp_enabled = self.plasticity_mode in ["stdp", "both"]
        if not stdp_enabled:
            return False
        
        if mode is None:
            mode = getattr(self, 'mode', LearningMode.ONLINE)
        
        return mode in [LearningMode.ONLINE, LearningMode.BATCH]
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass com STDP VETORIZADO."""
        output = super().forward(x, **kwargs)
        
        mode = kwargs.get('mode', getattr(self, 'mode', LearningMode.ONLINE))
        stdp_applied = self._should_apply_stdp(mode)
        
        if stdp_applied:
            self._update_stdp_traces(
                x, 
                output['spikes'],
                dt=kwargs.get('dt', 1.0)
            )
        
        # Métricas
        if self.trace_pre is not None:
            output['trace_pre_mean'] = self.trace_pre.mean()
            output['trace_post_mean'] = self.trace_post.mean()
        
        output['stdp_applied'] = torch.tensor(
            stdp_applied, 
            device=output['spikes'].device
        )
        
        return output
