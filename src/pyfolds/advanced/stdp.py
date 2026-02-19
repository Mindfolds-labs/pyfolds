"""Mixin para STDP (Spike-Timing Dependent Plasticity)."""

import math
import torch
from typing import Dict, Optional
from ..utils.types import LearningMode


class STDPMixin:
    """Mixin de STDP (*Spike-Timing Dependent Plasticity*) vetorizado.

    O mecanismo mantém traços pré e pós-sinápticos por amostra de batch e
    calcula atualizações locais de LTP/LTD sem loops em Python.

    Semântica implementada:

    - **LTP**: reforço proporcional ao traço pré quando há spike pós.
    - **LTD**: enfraquecimento proporcional ao traço pós quando há spike pós.

    Notes
    -----
    A atualização é aplicada ao tensor consolidado ``self.I`` quando disponível,
    com *clamp* para o intervalo ``[cfg.i_min, cfg.i_max]``.

    References
    ----------
    Bi, G. Q., & Poo, M. M. (1998).
    "Synaptic modifications in cultured hippocampal neurons".

    Semântica implementada (determinística):

    - Traços pré/pós são mantidos por amostra, com shape ``[B, D, S]``.
    - O spike pós-sináptico é global por amostra e é broadcast para
      todos os dendritos/sinapses.
    - Quando um neurônio dispara, todos os dendritos da mesma amostra
      recebem a mesma modulação plástica.

    Se for necessário STDP por dendrito específico, ``_update_stdp_traces``
    deve ser ajustado para aplicar máscara condicional por dendrito em vez
    do broadcast global.
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

        # Traços POR AMOSTRA
        self.trace_pre = None  # [B, D, S]
        self.trace_post = None  # [B, D, S]

    def _ensure_traces(self, batch_size: int, device: torch.device):
        """Garante a alocação dos traços com shape compatível com o batch.

        :param batch_size: Tamanho do batch corrente.
        :param device: Dispositivo onde os traços serão mantidos.
        """
        D = self.cfg.n_dendrites
        S = self.cfg.n_synapses_per_dendrite

        if self.trace_pre is None or self.trace_pre.shape[0] != batch_size:
            self.trace_pre = torch.zeros(batch_size, D, S, device=device)
            self.trace_post = torch.zeros(batch_size, D, S, device=device)

    def _update_stdp_traces(
        self, x: torch.Tensor, post_spike: torch.Tensor, dt: float = 1.0
    ):
        """Atualiza traços STDP e aplica deltas sinápticos vetorizados.

        :param x: Tensor de entrada com shape ``[B, D, S]``.
        :param post_spike: Spike pós-sináptico com shape ``[B]``.
        :param dt: Passo de tempo discreto da simulação.
        """
        batch_size = x.shape[0]
        device = x.device
        self._ensure_traces(batch_size, device)

        # Decaimento escalar
        decay_pre = math.exp(-dt / self.tau_pre)
        decay_post = math.exp(-dt / self.tau_post)

        # Decaimento (vetorizado)
        self.trace_pre.mul_(decay_pre)
        self.trace_post.mul_(decay_post)

        # Spikes pré (detectados por amostra)
        spike_threshold = getattr(self.cfg, "spike_threshold", 0.5)
        pre_spikes = (x > spike_threshold).float()  # [B, D, S]

        # Adiciona aos traços pré
        self.trace_pre.add_(pre_spikes)
        
        # Spike pós é global por amostra e broadcast para [B, 1, 1]
        # (mesma modulação para todos os dendritos/sinapses da amostra).
        post_expanded = post_spike.view(-1, 1, 1)  # [B, 1, 1]

        # LTD: onde trace_post > threshold
        trace_threshold = getattr(self.cfg, "stdp_trace_threshold", 0.01)
        ltd_mask = (self.trace_post > trace_threshold).float()
        delta_ltd = -self.A_minus * self.trace_post * ltd_mask * pre_spikes

        # LTP: onde trace_pre > threshold
        ltp_mask = (self.trace_pre > trace_threshold).float()
        delta_ltp = self.A_plus * self.trace_pre * ltp_mask * post_expanded

        # Aplica atualização diretamente nas sinapses reais.
        # self.I é uma visão consolidada/cached e não deve receber add_ in-place.
        if hasattr(self, "dendrites"):
            delta_total = (delta_ltd + delta_ltp).sum(dim=0)  # [D, S]
            with torch.no_grad():
                for d_idx, dend in enumerate(self.dendrites):
                    for s_idx, syn in enumerate(dend.synapses):
                        syn.I.add_(delta_total[d_idx, s_idx])
                        syn.I.clamp_(self.cfg.i_min, self.cfg.i_max)
                    dend._invalidate_cache()

        # Adiciona traço pós
        self.trace_post.add_(post_expanded)

    def _should_apply_stdp(self, mode: Optional[LearningMode] = None) -> bool:
        """Determina se STDP deve ser aplicado no passo atual.

        :param mode: Modo explícito de aprendizado. Se ``None``, usa
            ``self.mode`` quando disponível.
        :returns: ``True`` quando STDP está habilitado e o modo é compatível.
        """
        stdp_enabled = self.plasticity_mode in ["stdp", "both"]
        if not stdp_enabled:
            return False

        if mode is None:
            mode = getattr(self, "mode", LearningMode.ONLINE)

        return mode in [LearningMode.ONLINE, LearningMode.BATCH]

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Executa ``forward`` e acopla atualização de STDP quando habilitado.

        :param x: Entrada com shape ``[B, D, S]``.
        :param kwargs: Argumentos adicionais propagados ao ``forward`` base.
            Chaves relevantes:

            - ``mode``: modo de aprendizado.
            - ``dt``: passo temporal para o decaimento dos traços.

        :returns: Dicionário de saída do neurônio acrescido de métricas STDP.

        Example
        -------
        >>> import torch
        >>> class _Base:
        ...     def __init__(self):
        ...         self.mode = LearningMode.ONLINE
        ...     def forward(self, x, **kwargs):
        ...         return {"spikes": torch.ones(x.shape[0])}
        >>> class _Cfg:
        ...     n_dendrites = 1
        ...     n_synapses_per_dendrite = 2
        ...     i_min = 0.0
        ...     i_max = 1.0
        ...     spike_threshold = 0.0
        ...     stdp_trace_threshold = 0.0
        >>> class _Neuron(STDPMixin, _Base):
        ...     def __init__(self):
        ...         _Base.__init__(self)
        ...         self.cfg = _Cfg()
        ...         self.I = torch.zeros(1, 2)
        ...         self._init_stdp(plasticity_mode="stdp")
        >>> n = _Neuron()
        >>> out = n.forward(torch.ones(1, 1, 2))
        >>> bool(out["stdp_applied"].item())
        True
        >>> n.I.shape
        torch.Size([1, 2])
        """
        x_pre_stp = kwargs.pop("_x_pre_stp", x)
        output = super().forward(x, **kwargs)

        mode = kwargs.get("mode", getattr(self, "mode", LearningMode.ONLINE))
        stdp_applied = self._should_apply_stdp(mode)

        if stdp_applied:
            self._update_stdp_traces(x_pre_stp, output["spikes"], dt=kwargs.get("dt", 1.0))

        # Métricas
        if self.trace_pre is not None:
            output["trace_pre_mean"] = self.trace_pre.mean()
            output["trace_post_mean"] = self.trace_post.mean()

        output["stdp_applied"] = torch.tensor(
            stdp_applied, device=output["spikes"].device
        )

        return output
