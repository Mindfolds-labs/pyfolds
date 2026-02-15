"""MPJRDNeuronV2 - integração dendrítica cooperativa (Soft-WTA)."""

from typing import Optional, Dict

import torch

from .neuron import MPJRDNeuron
from ..utils.types import LearningMode


class MPJRDNeuronV2(MPJRDNeuron):
    """Versão experimental com soma cooperativa não-linear entre dendritos."""

    def forward(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        mode: Optional[LearningMode] = None,
        collect_stats: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass com integração cooperativa.

        x: [batch, n_dendrites, n_synapses]
        """
        effective_mode = mode if mode is not None else self.mode

        device = self.theta.device
        x = x.to(device)
        B, D, S = x.shape

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

        # 1) Pesos estruturais (Bartol log law)
        W = torch.log2(1.0 + self.N.float()) / self.cfg.w_scale

        # 2) Potenciais dendríticos vetorizados
        v_dend = torch.einsum("ds,bds->bd", W, x)

        # 3) Não-linearidade local por dendrito
        dendritic_gain = torch.sigmoid(v_dend - (self.theta * 0.5))

        # 4) Soma cooperativa
        somatic = dendritic_gain.sum(dim=-1)

        # 5) Disparo
        spikes = (somatic >= self.theta).float()

        spike_rate = spikes.mean().item()
        saturation_ratio = (self.N == self.cfg.n_max).float().mean().item()

        if effective_mode != LearningMode.INFERENCE and collect_stats:
            self.homeostasis.update(spike_rate)

        if self.cfg.neuromod_mode == "external":
            R_val = float(reward) if reward is not None else 0.0
            R_val = max(-1.0, min(1.0, R_val))
        else:
            R_val = self._compute_R_endogenous(spike_rate, saturation_ratio)
        R_tensor = torch.tensor([R_val], device=device)

        if collect_stats and effective_mode == LearningMode.BATCH and self.cfg.defer_updates:
            # Reuso do acumulador: "gated" agora representa ganho cooperativo médio.
            self.stats_acc.accumulate(x.detach(), dendritic_gain.detach(), spikes.detach())

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

        self.logger.trace(
            f"ForwardV2: batch={B}, spikes={spike_rate:.3f}, "
            f"θ={self.theta.item():.3f}, sat={saturation_ratio:.1%}"
        )

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
        }

    def _telemetry_forward_event(
        self,
        sid: int,
        spike_rate: float,
        saturation_ratio: float,
        R_tensor: torch.Tensor,
    ):
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
