"""Controle de homeostase do neurônio MPJRD - VERSÃO CORRIGIDA FINAL"""

import torch
import torch.nn as nn
from typing import Union, Optional
from .config import MPJRDConfig


class HomeostasisController(nn.Module):
    """
    Controla a homeostase do neurônio MPJRD.
    
    ✅ CORRIGIDO:
        - Removido +self.eps incorreto da média móvel
        - ✅ is_stable AGORA É PROPRIEDADE SEM PARÂMETROS
        - ✅ Adicionado is_stable_with_tolerance para custom tolerance
    """

    def __init__(self, cfg: MPJRDConfig):
        super().__init__()
        self.cfg = cfg
        
        self.register_buffer("theta", torch.tensor([cfg.theta_init]))
        self.register_buffer("r_hat", torch.tensor([cfg.target_spike_rate]))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        
        self.dead_neuron_threshold = cfg.dead_neuron_threshold
        self.dead_neuron_penalty_factor = cfg.dead_neuron_penalty
        self.activity_threshold = cfg.activity_threshold
        self.eps = cfg.homeostasis_eps

    def update(self, 
               current_rate: Union[float, torch.Tensor], 
               clamp_theta: bool = True) -> torch.Tensor:
        """Atualiza parâmetros homeostáticos."""
        # Validação com tolerância
        rate = float(current_rate) if isinstance(current_rate, torch.Tensor) else current_rate
        
        if rate < -self.eps or rate > 1.0 + self.eps:
            raise ValueError(f"current_rate deve estar em [0, 1], mas é {rate}")
        
        rate = max(0.0, min(1.0, rate))
        cfg = self.cfg

        # 1. Erro homeostático
        error = rate - cfg.target_spike_rate

        # 2. Ajuste do limiar
        delta_theta = cfg.homeostasis_eta * error
        self.theta.add_(delta_theta)

        # 3. Mecanismo de resgate
        if rate < self.dead_neuron_threshold:
            rescue_delta = -cfg.homeostasis_eta * self.dead_neuron_penalty_factor
            self.theta.add_(rescue_delta)

        # 4. Clamping
        if clamp_theta:
            self.theta.clamp_(cfg.theta_min, cfg.theta_max)

        # 5. Média móvel exponencial
        # ✅ CORRETO: r_hat = α * rate + (1-α) * r_hat
        # ❌ SEM o +self.eps incorreto!
        rate_tensor = torch.tensor([rate], device=self.theta.device)
        self.r_hat.mul_(1 - cfg.homeostasis_alpha).add_(
            cfg.homeostasis_alpha * rate_tensor
        )

        # 6. Contador
        self.step_count.add_(1)

        return self.theta

    def reset(self) -> None:
        """Reseta buffers."""
        self.theta.fill_(self.cfg.theta_init)
        self.r_hat.fill_(self.cfg.target_spike_rate)
        self.step_count.zero_()

    @property
    def homeostasis_error(self) -> torch.Tensor:
        """Erro homeostático atual (r_hat - target)."""
        return self.r_hat - self.cfg.target_spike_rate

    @property
    def is_stable(self) -> bool:
        """
        Verifica se homeostase está estável (tolerance padrão = 0.05).
        
        Returns:
            True se |r_hat - target| < 0.05
        """
        return abs(self.homeostasis_error.item()) < 0.05

    def is_stable_with_tolerance(self, tolerance: float = 0.05) -> bool:
        """
        Verifica estabilidade com tolerance customizado.
        
        Args:
            tolerance: Tolerância para considerar estabilidade
            
        Returns:
            True se |r_hat - target| < tolerance
        """
        return abs(self.homeostasis_error.item()) < tolerance

    def extra_repr(self) -> str:
        """Representação string detalhada do módulo."""
        return (f"θ={self.theta.item():.3f} "
                f"[{self.cfg.theta_min:.1f}, {self.cfg.theta_max:.1f}], "
                f"r̂={self.r_hat.item():.3f}, "
                f"target={self.cfg.target_spike_rate:.2f}, "
                f"resgate_th={self.dead_neuron_threshold:.2f}, "
                f"estável={self.is_stable}")