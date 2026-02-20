"""Mixin para dinâmica de curto prazo (STP - Short-Term Plasticity)."""

import math
import torch
from typing import Dict


class ShortTermDynamicsMixin:
    """
    Mixin para dinâmica de curto prazo (facilitação/depressão).
    
    ✅ SEMÂNTICA: GLOBAL (agregado sobre batch)
        - Estados u_stp e R_stp são COMPARTILHADOS [D, S]
        - Atualizações baseadas na MÉDIA do batch
        - Modulação aplicada igualmente a todas as amostras
    
    Implementa:
        - Facilitação (u): aumenta com spikes recentes
        - Depressão (R): diminui com spikes recentes (recursos)
    
    A dinâmica de curto prazo é TEMPORÁRIA (100-1000ms) e NÃO deve
    modificar os pesos estruturais (N) permanentemente.
    
    Baseado em:
        - Markram et al. (1998) - Differential signaling via the same axon
        - Tsodyks & Markram (1997) - The neural code between neocortical pyramidal neurons
    """
    
    def _init_short_term(self, u0: float = 0.1, R0: float = 1.0,
                          U: float = 0.2, tau_fac: float = 100.0,
                          tau_rec: float = 800.0):
        """
        Inicializa parâmetros de curto prazo.
        
        Args:
            u0: Facilitação inicial
            R0: Recursos iniciais
            U: Incremento de facilitação por spike
            tau_fac: Constante de tempo da facilitação (ms)
            tau_rec: Constante de tempo da recuperação (ms)
        """
        self.U = U
        self.tau_fac = tau_fac
        self.tau_rec = tau_rec
        
        D, S = self.cfg.n_dendrites, self.cfg.n_synapses_per_dendrite
        
        self.register_buffer("u_stp", torch.full((D, S), u0))
        self.register_buffer("R_stp", torch.full((D, S), R0))
    
    def _update_short_term_dynamics(self, x: torch.Tensor, dt: float = 1.0):
        """
        Atualiza variáveis de curto prazo baseado em spikes pré-sinápticos.
        
        ✅ CORRIGIDO:
            - Usa math.exp em vez de torch.tensor (zero overhead)
            - Operações mantidas no device correto via broadcasting
        
        Args:
            x: Tensor de entrada [B, D, S]
            dt: Passo de tempo (ms)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Esperado entrada com 3 dimensões [B, D, S], recebido shape {tuple(x.shape)}"
            )

        _, D, S = x.shape
        if D != self.cfg.n_dendrites or S != self.cfg.n_synapses_per_dendrite:
            raise ValueError(
                "Shape incompatível para dinâmica de curto prazo: "
                f"esperado [B, {self.cfg.n_dendrites}, {self.cfg.n_synapses_per_dendrite}], "
                f"recebido {tuple(x.shape)}"
            )

        # Garante buffers no mesmo device da entrada sem quebrar o registro
        # de buffers do nn.Module (state_dict/multi-GPU).
        if self.u_stp.device != x.device or self.R_stp.device != x.device:
            self.to(x.device)

        # Detecta spikes pré com threshold configurável
        spike_threshold = getattr(self.cfg, 'spike_threshold', 0.5)
        pre_spikes = (x > spike_threshold).float().mean(dim=0)  # [D, S] - média no batch
        
        # ✅ CORRIGIDO: decaimento escalar com math.exp
        decay_fac = math.exp(-dt / self.tau_fac)
        decay_rec = math.exp(-dt / self.tau_rec)
        
        # Atualiza facilitação (u)
        # u = u_prev * decay + U * (1 - u_prev) * pre_spikes
        u_prev = self.u_stp.clone()
        self.u_stp.copy_(u_prev * decay_fac + self.U * (1 - u_prev) * pre_spikes)
        self.u_stp.clamp_(0.0, 1.0)
        
        # Atualiza recursos (R)
        # ✅ Usa R_prev em TODOS os termos para manter consistência da discretização
        # R = R_prev * decay + (1 - R_prev) * (1 - decay) - u * R_prev * pre_spikes
        R_prev = self.R_stp.clone()
        recovery = (1 - R_prev) * (1 - decay_rec)
        depression = self.u_stp * R_prev * pre_spikes
        self.R_stp.copy_(R_prev * decay_rec + recovery - depression)
        
        self.R_stp.clamp_(0.0, 1.0)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass com dinâmica de curto prazo.
        
        ✅ SEMÂNTICA GLOBAL (agregado sobre batch)
        ✅ CORRIGIDO: sem criação de tensores desnecessários
        
        Args:
            x: Tensor de entrada [B, D, S]
            **kwargs: Argumentos adicionais (dt, etc.)
        
        Returns:
            Dict com spikes, potenciais e métricas de STP
        """
        dt = kwargs.get('dt', 1.0)
        
        # Atualiza dinâmica de curto prazo
        self._update_short_term_dynamics(x, dt)
        
        # ✅ MODULA ENTRADA (não os pesos!)
        # modulation: [D, S] → [1, D, S] para broadcasting
        modulation = self.u_stp * self.R_stp  # [D, S]
        x_modulated = x * modulation.unsqueeze(0)  # [B, D, S]
        
        # Forward da próxima classe com entrada modulada
        output = super().forward(x_modulated, _x_pre_stp=x, **kwargs)  # type: ignore
        
        # Adiciona métricas de STP ao output
        output['u_stp_mean'] = self.u_stp.mean()
        output['R_stp_mean'] = self.R_stp.mean()
        output['modulation_mean'] = modulation.mean()
        
        return output
