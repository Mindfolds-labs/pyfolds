"""Acumulador de estatísticas para batch learning - VERSÃO OTIMIZADA"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, Deque
from dataclasses import dataclass, field
import warnings
from collections import deque


@dataclass
class AccumulatedStats:
    """Container para estatísticas acumuladas."""
    x_mean: Optional[torch.Tensor] = None
    gated_mean: Optional[torch.Tensor] = None
    post_rate: Optional[float] = None
    v_dend_mean: Optional[torch.Tensor] = None
    u_mean: Optional[float] = None
    theta_mean: Optional[float] = None
    r_hat_mean: Optional[float] = None
    adaptation_mean: Optional[float] = None
    spike_count: int = 0
    total_samples: int = 0
    sparsity: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Verifica se há dados válidos."""
        return self.total_samples > 0
    
    def __repr__(self) -> str:
        """Representação legível."""
        if not self.is_valid():
            return "AccumulatedStats(empty)"
        return (f"AccumulatedStats(samples={self.total_samples}, "
                f"spikes={self.spike_count}, rate={self.post_rate:.3f})")


class StatisticsAccumulator(nn.Module):
    """
    Acumulador de estatísticas para aprendizado em batch.
    
    ✅ OTIMIZADO:
        - Uso de buffers in-place
        - Operações vetorizadas
        - Histórico com lazy evaluation
    """

    def __init__(self, 
                 n_dendrites: int, 
                 n_synapses: int, 
                 eps: float = 1e-8,
                 track_extra: bool = False,
                 max_history_len: int = 10000):
        super().__init__()
        
        self.n_dendrites = n_dendrites
        self.n_synapses = n_synapses
        self.eps = eps
        self.track_extra = track_extra
        self.max_history_len = max_history_len

        # Buffers principais
        self.register_buffer("acc_x", torch.zeros(n_dendrites, n_synapses))
        self.register_buffer("acc_gated", torch.zeros(n_dendrites))
        self.register_buffer("acc_spikes", torch.zeros(1, dtype=torch.long))
        self.register_buffer("acc_count", torch.zeros(1, dtype=torch.long))
        
        # Buffers extras
        if track_extra:
            self.register_buffer("acc_v_dend", torch.zeros(n_dendrites))
            self.register_buffer("acc_u", torch.zeros(1))
            self.register_buffer("acc_theta", torch.zeros(1))
            self.register_buffer("acc_r_hat", torch.zeros(1))
            self.register_buffer("acc_adaptation", torch.zeros(1))
        
        self.register_buffer("initialized", torch.tensor(False))
        
        # Histórico (lazy evaluation)
        self._history: Dict[str, Deque[float]] = {}
        self._history_enabled = False

    @property
    def history(self) -> Dict[str, Deque[float]]:
        """Histórico (deques circulares, criado sob demanda)."""
        if not self._history_enabled:
            return {}
        return self._history

    def enable_history(self, enable: bool = True) -> None:
        """Ativa/desativa histórico."""
        self._history_enabled = enable
        if enable and not self._history:
            self._history = {
                'spike_rate': deque(maxlen=self.max_history_len),
                'sparsity': deque(maxlen=self.max_history_len),
                'theta': deque(maxlen=self.max_history_len),
                'r_hat': deque(maxlen=self.max_history_len)
            }

    def reset(self) -> None:
        """Reseta todos os acumuladores."""
        self.acc_x.zero_()
        self.acc_gated.zero_()
        self.acc_spikes.zero_()
        self.acc_count.zero_()
        
        if self.track_extra:
            self.acc_v_dend.zero_()
            self.acc_u.zero_()
            self.acc_theta.zero_()
            self.acc_r_hat.zero_()
            self.acc_adaptation.zero_()
        
        self.initialized.fill_(False)
        
        if self._history_enabled:
            for key in self._history:
                self._history[key].clear()

    def accumulate(self, 
                  x: torch.Tensor, 
                  gated: torch.Tensor, 
                  spikes: torch.Tensor,
                  v_dend: Optional[torch.Tensor] = None,
                  u: Optional[torch.Tensor] = None,
                  theta: Optional[torch.Tensor] = None,
                  r_hat: Optional[torch.Tensor] = None,
                  adaptation: Optional[torch.Tensor] = None) -> None:
        """Acumula estatísticas de um batch (versão otimizada)."""
        # Validação rápida
        batch_size = x.shape[0]
        
        if x.shape[1:] != (self.n_dendrites, self.n_synapses):
            raise ValueError(
                f"Esperado [B, {self.n_dendrites}, {self.n_synapses}], "
                f"recebido {x.shape}"
            )

        # Device sync (só uma vez)
        device = x.device
        if self.acc_x.device != device:
            self.to(device)

        if not self.initialized.item():
            # Primeira batch
            self.acc_x.copy_(x.sum(dim=0))
            self.acc_gated.copy_(gated.sum(dim=0))
            self.acc_spikes.copy_(spikes.sum().long())
            self.acc_count.copy_(torch.tensor(batch_size, device=device, dtype=torch.long))
            
            if self.track_extra:
                if v_dend is not None:
                    self.acc_v_dend.copy_(v_dend.sum(dim=0))
                if u is not None:
                    self.acc_u.copy_(u.sum())
                if theta is not None:
                    self.acc_theta.copy_(theta.sum() if theta.dim() > 0 else theta * batch_size)
                if r_hat is not None:
                    self.acc_r_hat.copy_(r_hat.sum() if r_hat.dim() > 0 else r_hat * batch_size)
                if adaptation is not None:
                    self.acc_adaptation.copy_(adaptation.sum() if adaptation.dim() > 0 else adaptation * batch_size)
            
            self.initialized.fill_(True)
        else:
            # Acumulação vetorizada
            self.acc_x += x.sum(dim=0)
            self.acc_gated += gated.sum(dim=0)
            self.acc_spikes += spikes.sum().long()
            self.acc_count += batch_size
            
            if self.track_extra:
                if v_dend is not None:
                    self.acc_v_dend += v_dend.sum(dim=0)
                if u is not None:
                    self.acc_u += u.sum()
                if theta is not None:
                    self.acc_theta += theta.sum() if theta.dim() > 0 else theta * batch_size
                if r_hat is not None:
                    self.acc_r_hat += r_hat.sum() if r_hat.dim() > 0 else r_hat * batch_size
                if adaptation is not None:
                    self.acc_adaptation += adaptation.sum() if adaptation.dim() > 0 else adaptation * batch_size

        if self._history_enabled:
            self._update_history(spikes, gated, theta, r_hat)

    def _update_history(self, 
                       spikes: torch.Tensor, 
                       gated: torch.Tensor,
                       theta: Optional[torch.Tensor],
                       r_hat: Optional[torch.Tensor]) -> None:
        """Atualiza histórico (lazy)."""
        if not self._history_enabled:
            return
            
        batch_size = spikes.shape[0]
        spike_rate = spikes.float().mean().item()
        sparsity = (gated > 0).float().mean().item()
        
        self._history['spike_rate'].append(spike_rate)
        self._history['sparsity'].append(sparsity)
        
        if theta is not None:
            theta_mean = theta.mean().item() if theta.dim() > 0 else theta
            self._history['theta'].append(theta_mean)
        
        if r_hat is not None:
            r_hat_mean = r_hat.mean().item() if r_hat.dim() > 0 else r_hat
            self._history['r_hat'].append(r_hat_mean)

    def get_averages(self) -> AccumulatedStats:
        """Retorna médias de todas as estatísticas acumuladas."""
        stats = AccumulatedStats()
        
        if not self.has_data:
            return stats
        
        count = self.acc_count.float() + self.eps
        
        stats.x_mean = self.acc_x / count
        stats.gated_mean = self.acc_gated / count
        stats.post_rate = (self.acc_spikes.float() / count).item()
        stats.total_samples = int(self.acc_count.item())
        stats.spike_count = int(self.acc_spikes.item())
        
        if stats.gated_mean is not None:
            stats.sparsity = (stats.gated_mean > 0).float().mean().item()
        
        if self.track_extra:
            if hasattr(self, 'acc_v_dend'):
                stats.v_dend_mean = self.acc_v_dend / count
            if hasattr(self, 'acc_u'):
                stats.u_mean = (self.acc_u / count).item()
            if hasattr(self, 'acc_theta'):
                stats.theta_mean = (self.acc_theta / count).item()
            if hasattr(self, 'acc_r_hat'):
                stats.r_hat_mean = (self.acc_r_hat / count).item()
            if hasattr(self, 'acc_adaptation'):
                stats.adaptation_mean = (self.acc_adaptation / count).item()
        
        return stats

    def plot_history(self, keys: Optional[List[str]] = None, figsize=(10, 6)):
        """Plota histórico (retorna figura para maior controle)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib não disponível para plotagem")
            return None
        
        if not self._history_enabled or not self._history:
            print("Histórico não ativado. Use enable_history(True) primeiro.")
            return None
        
        if keys is None:
            keys = ['spike_rate', 'sparsity', 'theta', 'r_hat']
        
        available_keys = [k for k in keys if k in self._history and self._history[k]]
        
        if not available_keys:
            print("Nenhum dado histórico disponível")
            return None
        
        n_plots = len(available_keys)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        for ax, key in zip(axes, available_keys):
            data = self._history[key]
            ax.plot(data)
            ax.set_title(f'{key.replace("_", " ").title()} over Time')
            ax.set_xlabel('Batch')
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig  # ✅ Retorna figura em vez de mostrar

    @property
    def has_data(self) -> bool:
        return self.initialized.item() and self.acc_count.item() > 0

    @property
    def batch_count(self) -> int:
        return int(self.acc_count.item()) if self.has_data else 0

    def extra_repr(self) -> str:
        status = "active" if self.has_data else "empty"
        history = "on" if self._history_enabled else "off"
        return (f"D={self.n_dendrites}, S={self.n_synapses}, "
                f"batches={self.batch_count}, status={status}, "
                f"track_extra={self.track_extra}, history={history}")


def create_accumulator_from_config(config, track_extra: bool = False) -> StatisticsAccumulator:
    """Cria accumulator a partir de config."""
    return StatisticsAccumulator(
        n_dendrites=config.n_dendrites,
        n_synapses=config.n_synapses_per_dendrite,
        eps=config.eps,
        track_extra=track_extra
    )