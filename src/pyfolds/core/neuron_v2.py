"""MPJRDNeuronV2 - integração dendrítica cooperativa (Soft-WTA)."""

from typing import Optional, Dict

import torch

from .neuron import MPJRDNeuron
from ..utils.types import LearningMode


class MPJRDNeuronV2(MPJRDNeuron):
    """
    Versão experimental com soma cooperativa não-linear entre dendritos.
    
    ✅ CORRIGIDO:
        - Herda todas as melhorias do MPJRDNeuron (validação de devices, logging, etc.)
        - Reutiliza validação de input device da classe base
        - Mantém consistência com a versão base
        - Preserva semântica do acumulador
    """

    def forward(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        mode: Optional[LearningMode] = None,
        collect_stats: bool = True,
        dt: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass com integração cooperativa.

        Args:
            x: Tensor de entrada [batch, dendrites, synapses]
            reward: Sinal de recompensa externo
            mode: Modo de aprendizado (sobrescreve o atual)
            collect_stats: Se deve coletar estatísticas
            dt: Passo de tempo (ms)

        Returns:
            Dict com spikes, potenciais e estatísticas
        """
        effective_mode = mode if mode is not None else self.mode
        
        # ✅ Valida device (herdado da classe base)
        self._validate_input_device(x)
        
        device = self.theta.device
        B, D, S = x.shape

        # Valida dimensões
        if D != self.cfg.n_dendrites or S != self.cfg.n_synapses_per_dendrite:
            self.logger.error(
                "❌ Dimensão inválida: esperado (%s, %s), recebido (%s, %s)",
                self.cfg.n_dendrites,
                self.cfg.n_synapses_per_dendrite,
                D,
                S,
            )
            raise ValueError(
                f"Esperado ({self.cfg.n_dendrites}, {self.cfg.n_synapses_per_dendrite}), "
                f"recebido ({D}, {S})"
            )

        # ===== 1. PESOS ESTRUTURAIS (Bartol log law) =====
        # Usa propriedade N da classe base (já está no device correto)
        W = torch.log2(1.0 + self.N.float()) / self.cfg.w_scale  # [D, S]

        # ===== 2. POTENCIAIS DENDRÍTICOS VETORIZADOS =====
        # v_dend = Σ(W * x) por dendrito
        v_dend = torch.einsum("ds,bds->bd", W, x)  # [B, D]

        # ===== 3. NÃO-LINEARIDADE LOCAL POR DENDRITO =====
        # Ganho dendrítico = sigmoid(v_dend - θ/2)
        dendritic_gain = torch.sigmoid(v_dend - (self.theta * 0.5))  # [B, D]

        # ===== 4. SOMA COOPERATIVA =====
        somatic = dendritic_gain.sum(dim=-1)  # [B]

        # ===== 5. DISPARO =====
        spikes = (somatic >= self.theta).float()  # [B]

        # ===== 6. ESTATÍSTICAS =====
        spike_rate = spikes.mean().item()
        saturation_ratio = (self.N == self.cfg.n_max).float().mean().item()

        # ===== 7. HOMEOSTASE =====
        if effective_mode != LearningMode.INFERENCE and collect_stats:
            self.homeostasis.update(spike_rate)

        # ===== 8. NEUROMODULAÇÃO =====
        if self.cfg.neuromod_mode == "external":
            R_val = float(reward) if reward is not None else 0.0
            R_val = max(-1.0, min(1.0, R_val))
        else:
            R_val = self._compute_R_endogenous(spike_rate, saturation_ratio)
        R_tensor = torch.tensor([R_val], device=device)

        # ===== 9. ACUMULAÇÃO (BATCH MODE) =====
        if collect_stats and effective_mode == LearningMode.BATCH and self.cfg.defer_updates:
            # Acumula para atualização posterior
            self.stats_acc.accumulate(x.detach(), dendritic_gain.detach(), spikes.detach())

        # ===== 9b. ATUALIZAÇÃO IMEDIATA (ONLINE) =====
        if (
            collect_stats
            and effective_mode == LearningMode.ONLINE
            and self.cfg.plastic
            and not self.cfg.defer_updates
        ):
            # Aplica plasticidade online (implementado na classe base)
            self._apply_online_plasticity(
                x=x.detach(),
                post_rate=spike_rate,
                R_tensor=R_tensor,
                dt=dt,
                mode=effective_mode,
            )

        # ===== 10. TELEMETRIA =====
        self.step_id.add_(1)
        if self.telemetry is not None and self.telemetry.enabled():
            sid = int(self.step_id.item())
            self.telemetry.emit(
                self._telemetry_forward_event(
                    sid=sid,
                    spike_rate=spike_rate,
                    saturation_ratio=saturation_ratio,
                    R_tensor=R_tensor,
                )
            )

        # ===== 11. LOGGING =====
        self.logger.trace(
            f"ForwardV2: batch={B}, spikes={spike_rate:.3f}, "
            f"θ={self.theta.item():.3f}, sat={saturation_ratio:.1%}"
        )
        
        if spike_rate < self.cfg.dead_neuron_threshold:
            self.logger.warning(
                f"⚠️ Neurônio V2 hipoativo! rate={spike_rate:.3f} "
                f"(threshold={self.cfg.dead_neuron_threshold})"
            )
        
        if saturation_ratio > 0.5:
            self.logger.info(f"📊 Saturação alta (V2): {saturation_ratio:.1%}")
        
        if R_val > 0.8:
            self.logger.debug(f"🎯 Neuromodulação alta (V2): R={R_val:.2f}")

        # ===== 12. RETORNO =====
        return {
            "spikes": spikes,
            "u": somatic,
            "somatic": somatic,
            "v_dend": v_dend,
            "gated": dendritic_gain,
            "dendritic_gain": dendritic_gain,
            "theta": self.theta.clone(),
            "r_hat": self.r_hat.clone(),
            "spike_rate": torch.tensor(spike_rate, device=device),
            "saturation_ratio": torch.tensor(saturation_ratio, device=device),
            "R": R_tensor,
            "N_mean": self.N.float().mean().to(device),
            "W_mean": self.W.float().mean().to(device),
            "I_mean": self.I.float().mean().to(device),
            "mode": self.mode.value,
            "device": str(device),
        }

    def _telemetry_forward_event(
        self,
        sid: int,
        spike_rate: float,
        saturation_ratio: float,
        R_tensor: torch.Tensor,
    ):
        """Gera evento de telemetria para forward (compatível com versão base)."""
        from ..telemetry import forward_event

        return forward_event(
            step_id=sid,
            mode=self.mode.value,
            spike_rate=spike_rate,
            theta=float(self.theta.item()),
            r_hat=float(self.r_hat.item()),
            saturation_ratio=saturation_ratio,
            R=float(R_tensor.item()),
            N_mean=float(self.N.float().mean().item()),
            I_mean=float(self.I.float().mean().item()),
            W_mean=float(self.W.float().mean().item()),
        )

    def extra_repr(self) -> str:
        """Representação string."""
        return (f"V2 - mode={self.mode.value}, D={self.cfg.n_dendrites}, "
                f"S={self.cfg.n_synapses_per_dendrite}, θ={self.theta.item():.2f}, "
                f"device={self.theta.device}")