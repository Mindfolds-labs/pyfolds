"""Dendrito MPJRD - Agregação vetorizada de sinapses - VERSÃO OTIMIZADA"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .config import MPJRDConfig
from .synapse import MPJRDSynapse


class MPJRDDendrite(nn.Module):
    """
    Dendrito MPJRD com processamento vetorizado e cache OTIMIZADO.
    
    ✅ OTIMIZAÇÃO CRÍTICA:
        - Cache único que coleta estados UMA ÚNICA VEZ
        - Evita loops múltiplos sobre as sinapses
        - Propriedades retornam do cache, não recalculam
    """
    
    def __init__(self, cfg: MPJRDConfig, dendrite_id: int):
        super().__init__()
        self.cfg = cfg
        self.id = dendrite_id
        self.n_synapses = cfg.n_synapses_per_dendrite
        
        # Sinapses como ModuleList
        self.synapses = nn.ModuleList([
            MPJRDSynapse(cfg) for _ in range(self.n_synapses)
        ])
        
        # Cache único (dicionário)
        self._cached_states = None
        self._cache_invalid = True

    def _ensure_cache_valid(self):
        """
        ✅ OTIMIZADO: Coleta estados UMA ÚNICA VEZ e cacheia tudo.
        Propriedades individuais retornam do cache, não recalculam.
        """
        if not self._cache_invalid and self._cached_states is not None:
            return
        
        first_synapse = self.synapses[0] if len(self.synapses) > 0 else None
        device = first_synapse.N.device if first_synapse is not None else torch.device(self.cfg.device)
        
        # ✅ ÚNICO LOOP sobre sinapses (coleta tudo de uma vez)
        N_list = []
        I_list = []
        # Nota técnica:
        # O núcleo MPJRD mantém apenas estados estruturais (N) e voláteis (I)
        # por sinapse. Estados de STP (u, R) pertencem ao módulo avançado
        # (ShortTermDynamicsMixin) em nível de neurônio, não ao core de sinapse.
        has_short_term_state = all(hasattr(syn, "u") and hasattr(syn, "R") for syn in self.synapses)
        u_list = [] if has_short_term_state else None
        R_list = [] if has_short_term_state else None
        
        for syn in self.synapses:
            N_list.append(syn.N.to(device))
            I_list.append(syn.I.to(device))
            if has_short_term_state:
                u_list.append(syn.u.to(device))
                R_list.append(syn.R.to(device))
        
        # ✅ Concatena de uma vez (operação vetorizada)
        self._cached_states = {
            'N': torch.cat(N_list).to(torch.int32),
            'I': torch.cat(I_list),
            **({'u': torch.cat(u_list), 'R': torch.cat(R_list)} if has_short_term_state else {})
        }
        
        # ✅ W é derivado de N (calculado sob demanda, cacheado)
        self._cached_W = torch.log2(1.0 + self._cached_states['N'].float()) / self.cfg.w_scale
        
        self._cache_invalid = False

    def _invalidate_cache(self):
        """Invalida o cache (chamar após modificar sinapses)."""
        self._cache_invalid = True
        self._cached_states = None
        self._cached_W = None

    @property
    def N(self) -> torch.Tensor:
        """Filamentos [S] - ✅ retorna do cache."""
        self._ensure_cache_valid()
        return self._cached_states['N']

    @property
    def I(self) -> torch.Tensor:
        """Potencial interno [S] - ✅ retorna do cache."""
        self._ensure_cache_valid()
        return self._cached_states['I']

    @property
    def u(self) -> Optional[torch.Tensor]:
        """Facilitação [S] quando disponível no backend sináptico."""
        self._ensure_cache_valid()
        return self._cached_states.get('u')

    @property
    def R(self) -> Optional[torch.Tensor]:
        """Recuperação [S] quando disponível no backend sináptico."""
        self._ensure_cache_valid()
        return self._cached_states.get('R')

    @property
    def W(self) -> torch.Tensor:
        """Pesos [S] derivados de N - ✅ cacheado."""
        self._ensure_cache_valid()
        if self._cached_W is None:
            self._cached_W = torch.log2(1.0 + self.N.float()) / self.cfg.w_scale
        return self._cached_W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass vetorizado."""
        # Normaliza entrada para [B, S]
        if x.dim() == 2:
            if x.shape[1] != self.n_synapses:
                raise ValueError(f"Esperado {self.n_synapses} sinapses, recebido {x.shape[1]}")
            x_flat = x
        elif x.dim() == 3:
            if x.shape[1] == 1:
                x_flat = x.squeeze(1)
            elif self.id < x.shape[1]:
                x_flat = x[:, self.id, :]
            else:
                raise ValueError(f"Dendrite id {self.id} fora do range {x.shape[1]}")
        else:
            raise ValueError(f"Esperado 2 ou 3 dimensões, recebido {x.dim()}")

        # Produto interno vetorizado
        weights = self.W.to(x_flat.device)  # [S]
        v_dend = torch.einsum('s,bs->b', weights, x_flat)  # [B]
        
        return v_dend

    @torch.no_grad()
    def update_synapses_rate_based(self, 
                                  pre_rate: torch.Tensor,
                                  post_rate: torch.Tensor,
                                  R: torch.Tensor,
                                  dt: float = 1.0,
                                  mode=None) -> None:
        """Atualiza sinapses de forma indexada (pré-sináptico por sinapse)."""
        if pre_rate.dim() == 2 and pre_rate.shape[1] == 1:
            pre_rate = pre_rate.squeeze(1)
        
        # Regra hebbiana local: cada sinapse deve receber sua taxa pré-sináptica
        # correspondente para preservar especificidade sináptica.
        if pre_rate.numel() not in (1, self.n_synapses):
            raise ValueError(
                f"pre_rate deve ter 1 ou {self.n_synapses} elementos, recebeu {pre_rate.numel()}"
            )

        for s_idx, syn in enumerate(self.synapses):
            local_pre = pre_rate if pre_rate.numel() == 1 else pre_rate[s_idx:s_idx + 1]
            syn.update(local_pre, post_rate, R, dt, mode)
        
        self._invalidate_cache()

    def consolidate(self, dt: float = 1.0) -> None:
        """Consolida mudanças sinápticas."""
        for syn in self.synapses:
            syn.consolidate(dt)
        self._invalidate_cache()

    def get_states(self) -> Dict[str, torch.Tensor]:
        """Retorna todos os estados."""
        self._ensure_cache_valid()
        states = {
            'N': self.N.clone(),
            'W': self.W.clone(),
            'I': self.I.clone(),
        }
        if self.u is not None and self.R is not None:
            states.update({'u': self.u.clone(), 'R': self.R.clone()})
        return states

    def load_states(self, states: Dict[str, torch.Tensor]) -> None:
        """Carrega estados."""
        for idx, syn in enumerate(self.synapses):
            if 'N' in states:
                syn.N.data = states['N'][idx].view(1)
            if 'I' in states:
                syn.I.data = states['I'][idx].view(1)
            if 'u' in states and hasattr(syn, 'u'):
                syn.u.data = states['u'][idx].view(1)
            if 'R' in states and hasattr(syn, 'R'):
                syn.R.data = states['R'][idx].view(1)
        self._invalidate_cache()

    def extra_repr(self) -> str:
        """Representação string."""
        if self._cached_states is None:
            return f"id={self.id}, synapses={self.n_synapses} (cache vazio)"
        
        return (f"id={self.id}, synapses={self.n_synapses}, "
                f"N_mean={self.N.float().mean().item():.1f}")
