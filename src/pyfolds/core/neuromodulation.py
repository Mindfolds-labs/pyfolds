# pyfolds/core/neuromodulation.py
"""
Neuromodulador para o neurônio MPJRD

Calcula o sinal neuromodulador R baseado em diferentes estratégias:
- 'external': R vem de recompensa externa (reward)
- 'capacity': R baseado em capacidade livre (saturação de N)
- 'surprise': R baseado na diferença entre taxa atual e média (|rate - r_hat|)

O sinal R é sempre clampado em [-1, 1] para estabilidade.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Union
from .config import MPJRDConfig


class Neuromodulator(nn.Module):
    """
    Neuromodulador para o neurônio MPJRD.
    
    Este módulo calcula o sinal neuromodulador R que influencia a plasticidade
    sináptica. Diferentes estratégias podem ser usadas, configuradas via
    `cfg.neuromod_mode`.
    
    Modos disponíveis:
        - 'external': R = reward (fornecido externamente)
        - 'capacity': R baseado em capacidade livre (1 - saturação)
        - 'surprise': R = bias + k * |rate - r_hat|
    
    Args:
        cfg: Configuração do neurônio (MPJRDConfig)
    
    Example:
        >>> cfg = MPJRDConfig(neuromod_mode='surprise', sup_k=2.0, sup_bias=0.0)
        >>> neuromod = Neuromodulator(cfg)
        >>> R = neuromod(rate=0.25, r_hat=0.10)
        >>> print(R.item())  # 0.30 (surpresa de 0.15 * 2.0)
    """
    
    def __init__(self, cfg: MPJRDConfig):
        super().__init__()
        self.cfg = cfg
    
    def forward(
        self, 
        rate: Union[float, torch.Tensor], 
        r_hat: Union[float, torch.Tensor],
        saturation_ratio: Optional[float] = None,
        reward: Optional[Union[float, torch.Tensor]] = None, 
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calcula o sinal neuromodulador R.
        
        Args:
            rate: Taxa de disparo atual [0, 1]
            r_hat: Média móvel da taxa [0, 1]
            saturation_ratio: Proporção de sinapses saturadas [0,1] (para modo 'capacity')
            reward: Recompensa externa (obrigatória no modo 'external')
            device: Device para o tensor resultante (se None, usa CPU)
        
        Returns:
            Tensor escalar R no device especificado, clampado em [-1, 1]
        
        Raises:
            ValueError: Se reward for None no modo 'external'
            ValueError: Se saturation_ratio for None no modo 'capacity'
        """
        if device is None:
            if isinstance(rate, torch.Tensor):
                device = rate.device
            elif isinstance(r_hat, torch.Tensor):
                device = r_hat.device
            elif isinstance(reward, torch.Tensor):
                device = reward.device
            else:
                device = torch.device("cpu")

        def _to_scalar(value: Union[float, torch.Tensor], name: str) -> float:
            if isinstance(value, torch.Tensor):
                scalar = value.item() if value.numel() == 1 else float(value.mean())
            else:
                scalar = float(value)
            if not math.isfinite(scalar):
                raise ValueError(f"{name} deve ser finito, obtido {scalar}")
            return scalar

        # Converte tensores para escalar e valida domínio numérico
        rate = _to_scalar(rate, "rate")
        r_hat = _to_scalar(r_hat, "r_hat")

        # Valida ranges
        rate = max(0.0, min(1.0, rate))
        r_hat = max(0.0, min(1.0, r_hat))
        
        # Calcula R baseado no modo configurado
        if self.cfg.neuromod_mode == "external":
            # Modo external: usa recompensa externa
            if reward is None:
                raise ValueError("reward must be provided when neuromod_mode='external'")
            
            reward = _to_scalar(reward, "reward")
            R_val = reward
            
        elif self.cfg.neuromod_mode == "capacity":
            # Modo capacity: baseado em capacidade livre (saturação)
            if saturation_ratio is None:
                raise ValueError("saturation_ratio must be provided when neuromod_mode='capacity'")
            saturation_ratio = float(saturation_ratio)
            if not math.isfinite(saturation_ratio):
                raise ValueError("saturation_ratio deve ser finito no modo 'capacity'")

            saturation_ratio = max(0.0, min(1.0, saturation_ratio))
            free_capacity = 1.0 - saturation_ratio

            # Penalidade por desvio da taxa alvo
            rate_error = abs(rate - self.cfg.target_spike_rate)
            if math.isnan(rate_error):
                raise ValueError(
                    f"rate_error resultou em NaN (rate={rate}, target={self.cfg.target_spike_rate})"
                )

            R_val = (
                self.cfg.cap_bias
                + self.cfg.cap_k_sat * free_capacity
                - self.cfg.cap_k_rate * rate_error
            )
            if math.isnan(R_val):
                raise ValueError("R_val resultou em NaN no modo 'capacity'")

            # Clamp intermediário para evitar explosões numéricas antes do clamp biológico final.
            R_val = max(-10.0, min(10.0, R_val))
            
        elif self.cfg.neuromod_mode == "surprise":
            # Modo surprise: baseado em erro de predição
            surprise = abs(rate - r_hat)
            R_val = self.cfg.sup_bias + self.cfg.sup_k * surprise
            
        else:
            # Modo desconhecido: neutro
            R_val = 0.0
        
        # Clamp para faixa biológica [-1, 1]
        R_val = max(-1.0, min(1.0, R_val))
        
        # Cria tensor no device correto
        return torch.tensor([R_val], device=device, dtype=torch.float32)
    
    def extra_repr(self) -> str:
        """Representação string do módulo."""
        return f"mode={self.cfg.neuromod_mode}"