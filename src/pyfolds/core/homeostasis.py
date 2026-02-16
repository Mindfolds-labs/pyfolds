"""Controle de homeostase do neurônio MPJRD - VERSÃO CORRIGIDA FINAL"""

import torch
import torch.nn as nn
from typing import Union, Callable
from .config import MPJRDConfig


class HomeostasisController(nn.Module):
    """
    Controla a homeostase do neurônio MPJRD.
    
    ✅ CORRIGIDO:
        - Removido +self.eps incorreto da média móvel
        - is_stable unificado como método com tolerance opcional
        - Adicionado callback para eventos de estabilidade
    """

    def __init__(self, cfg: MPJRDConfig):
        super().__init__()
        self.cfg = cfg
        
        self.register_buffer("theta", torch.tensor([cfg.theta_init]))
        self.register_buffer("r_hat", torch.tensor([cfg.target_spike_rate]))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("integral_error", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("last_error", torch.zeros(1, dtype=torch.float32))
        
        self.dead_neuron_threshold = cfg.dead_neuron_threshold
        self.dead_neuron_penalty_factor = cfg.dead_neuron_penalty
        self.activity_threshold = cfg.activity_threshold
        self.eps = cfg.homeostasis_eps
        
        # Callbacks para eventos
        self._on_stable_callbacks: list[Callable] = []
        self._on_unstable_callbacks: list[Callable] = []
        
        # Estado de estabilidade
        self._was_stable = False

        # Ganhos PID-like para estabilizar convergência
        self.kp = cfg.homeostasis_eta
        self.ki = cfg.homeostasis_eta * 0.1
        self.kd = cfg.homeostasis_eta * 0.01

    def update(self, 
               current_rate: Union[float, torch.Tensor], 
               clamp_theta: bool = True) -> torch.Tensor:
        """
        Atualiza parâmetros homeostáticos.
        
        Args:
            current_rate: Taxa de disparo atual [0, 1]
            clamp_theta: Se deve aplicar clamping no theta
            
        Returns:
            theta atualizado
        """
        # Validação com tolerância
        rate = float(current_rate) if isinstance(current_rate, torch.Tensor) else current_rate
        
        if rate < -self.eps or rate > 1.0 + self.eps:
            raise ValueError(f"current_rate deve estar em [0, 1], mas é {rate}")
        
        rate = max(0.0, min(1.0, rate))
        cfg = self.cfg

        # 1. Erro homeostático
        error = rate - cfg.target_spike_rate

        # 2. Ajuste robusto do limiar (PID-like)
        delta_p = self.kp * error

        self.integral_error.add_(error * cfg.dt)
        self.integral_error.clamp_(-1.0, 1.0)
        delta_i = self.ki * self.integral_error

        d_error = error - float(self.last_error.item())
        delta_d = self.kd * d_error / max(cfg.dt, 1e-6)
        self.last_error.fill_(error)

        delta_theta = torch.tensor([delta_p], device=self.theta.device)
        delta_theta.add_(delta_i).add_(delta_d)
        delta_theta.clamp_(-0.5, 0.5)
        self.theta.add_(delta_theta)

        # 3. Mecanismo de resgate
        if rate < self.dead_neuron_threshold:
            rescue_delta = -cfg.homeostasis_eta * self.dead_neuron_penalty_factor
            self.theta.add_(rescue_delta)

        # 4. Clamping
        if clamp_theta:
            self.theta.clamp_(cfg.theta_min, cfg.theta_max)

        # 5. Média móvel exponencial
        rate_tensor = torch.tensor([rate], device=self.theta.device)
        self.r_hat.mul_(1 - cfg.homeostasis_alpha).add_(
            cfg.homeostasis_alpha * rate_tensor
        )

        # 6. Verifica mudança de estado de estabilidade
        self._check_stability_change()

        # 7. Contador
        self.step_count.add_(1)

        return self.theta

    def _check_stability_change(self, tolerance: float = 0.05) -> None:
        """Verifica se houve mudança no estado de estabilidade."""
        is_stable_now = self.is_stable(tolerance)
        
        if is_stable_now and not self._was_stable:
            # Transição instável → estável
            for callback in self._on_stable_callbacks:
                callback(self)
        elif not is_stable_now and self._was_stable:
            # Transição estável → instável
            for callback in self._on_unstable_callbacks:
                callback(self)
        
        self._was_stable = is_stable_now

    def on_stable(self, callback: Callable[['HomeostasisController'], None]) -> None:
        """Registra callback para quando a homeostase se tornar estável."""
        self._on_stable_callbacks.append(callback)

    def on_unstable(self, callback: Callable[['HomeostasisController'], None]) -> None:
        """Registra callback para quando a homeostase se tornar instável."""
        self._on_unstable_callbacks.append(callback)

    def reset(self) -> None:
        """Reseta buffers."""
        self.theta.fill_(self.cfg.theta_init)
        self.r_hat.fill_(self.cfg.target_spike_rate)
        self.step_count.zero_()
        self.integral_error.zero_()
        self.last_error.zero_()
        self._was_stable = False

    @property
    def homeostasis_error(self) -> torch.Tensor:
        """Erro homeostático atual (r_hat - target)."""
        return self.r_hat - self.cfg.target_spike_rate

    def is_stable(self, tolerance: float = 0.05) -> bool:
        """Verifica se homeostase está estável com tolerância configurável."""
        return abs(self.homeostasis_error.item()) < tolerance

    def stability_ratio(self, window: int = 100) -> float:
        """
        Calcula proporção de passos estáveis na janela recente.
        
        Args:
            window: Número de passos para considerar
            
        Returns:
            Float entre 0 e 1 indicando proporção de estabilidade
        """
        if self.step_count.item() < window:
            return 0.0
        
        # Implementação simplificada - em produção usaria buffer circular
        return 1.0 if self.is_stable() else 0.0

    def extra_repr(self) -> str:
        """Representação string detalhada do módulo."""
        return (f"θ={self.theta.item():.3f} "
                f"[{self.cfg.theta_min:.1f}, {self.cfg.theta_max:.1f}], "
                f"r̂={self.r_hat.item():.3f}, "
                f"target={self.cfg.target_spike_rate:.2f}, "
                f"resgate_th={self.dead_neuron_threshold:.2f}, "
                f"estável={self.is_stable()}")
