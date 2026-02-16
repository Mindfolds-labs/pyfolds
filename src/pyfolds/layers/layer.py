"""Camada de neurônios MPJRD - VERSÃO FINAL CORRIGIDA"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any, Type
from ..core.config import MPJRDConfig
from ..core.neuron import MPJRDNeuron
from ..utils.types import LearningMode


class MPJRDLayer(nn.Module):
    """
    Camada de neurônios MPJRD para construção de redes neurais.
    
    Características:
        - Múltiplos neurônios em paralelo
        - Processamento em batch eficiente
        - Suporte a telemetria (repassada para neurônios)
        - Métricas agregadas por camada
        - Modos de aprendizado (online/batch/inference)
    
    Args:
        n_neurons: Número de neurônios na camada
        cfg: Configuração dos neurônios (compartilhada)
        name: Nome opcional para identificação
        enable_telemetry: Se True, ativa telemetria nos neurônios
        telemetry_profile: Perfil de telemetria ('off', 'light', 'heavy')
        device: Device para os tensores (opcional)
    
    Example:
        >>> cfg = MPJRDConfig(n_dendrites=4, n_synapses_per_dendrite=32)
        >>> layer = MPJRDLayer(10, cfg, name="hidden1", enable_telemetry=True)
        >>> x = torch.randn(128, 10, 4, 32)  # [batch, neurons, dendrites, synapses]
        >>> out = layer(x)
        >>> print(out['spikes'].shape)  # [128, 10]
    """

    def __init__(
        self,
        n_neurons: int,
        cfg: MPJRDConfig,
        name: str = "",
        neuron_cls: Type[MPJRDNeuron] = MPJRDNeuron,
        enable_telemetry: bool = False,
        telemetry_profile: str = "off",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.name = name or f"layer_{id(self)}"
        self.cfg = cfg
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        legacy_neuron_class = kwargs.pop("neuron_class", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Argumentos não reconhecidos: {unexpected}")

        if legacy_neuron_class is not None:
            if neuron_cls is not MPJRDNeuron and legacy_neuron_class is not neuron_cls:
                raise ValueError("Use apenas um parâmetro de classe de neurônio")
            neuron_cls = legacy_neuron_class

        if not issubclass(neuron_cls, MPJRDNeuron):
            raise TypeError("neuron_cls deve herdar de MPJRDNeuron")

        # Cria neurônios com telemetria (se ativada)
        self.neuron_cls = neuron_cls
        self.neurons = nn.ModuleList([
            neuron_cls(
                cfg,
                enable_telemetry=enable_telemetry,
                telemetry_profile=telemetry_profile,
                name=f"{self.name}.n{i}",
            )
            for i in range(n_neurons)
        ])

        # Move para device
        self.to(self.device)

    @property
    def n_dendrites(self) -> int:
        """Número de dendritos por neurônio (da config)."""
        return self.cfg.n_dendrites

    @property
    def n_synapses(self) -> int:
        """Número de sinapses por dendrito (da config)."""
        return self.cfg.n_synapses_per_dendrite

    @property
    def theta_mean(self) -> float:
        """Média dos thresholds de todos neurônios."""
        if not self.neurons:
            return 0.0
        # ✅ Mais eficiente: usa tensor e mean()
        thetas = torch.tensor([n.theta.item() for n in self.neurons])
        return thetas.mean().item()

    @property
    def r_hat_mean(self) -> float:
        """Média das taxas de todos neurônios."""
        if not self.neurons:
            return 0.0
        # ✅ Mais eficiente: usa tensor e mean()
        r_hats = torch.tensor([n.r_hat.item() for n in self.neurons])
        return r_hats.mean().item()

    def forward(
        self,
        x: torch.Tensor,
        reward: Optional[float] = None,
        mode: Optional[LearningMode] = None,
        neuron_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass da camada.
        
        Args:
            x: Tensor de entrada. Formatos aceitos:
                - [batch, n_neurons, n_dendrites, n_synapses] (completo)
                - [batch, n_neurons, n_dendrites] (expande sinapses)
                - [batch, n_dendrites, n_synapses] (mesmo para todos neurônios)
            reward: Sinal de recompensa (opcional)
            mode: Modo de aprendizado (opcional)
        
        Returns:
            Dict com:
                - spikes: [batch, n_neurons] spikes de cada neurônio
                - rates: [n_neurons] taxas médias
                - thetas: [n_neurons] thresholds atuais
                - r_hats: [n_neurons] taxas médias móveis
        """
        # Prepara entrada
        x = x.to(self.device)
        x = self._prepare_input(x)
        batch_size = x.shape[0]

        neuron_kwargs = neuron_kwargs or {}

        # Pré-aloca tensores de saída
        spikes = torch.zeros(batch_size, self.n_neurons, device=self.device)
        rates = torch.zeros(self.n_neurons, device=self.device)
        thetas = torch.zeros(self.n_neurons, device=self.device)
        r_hats = torch.zeros(self.n_neurons, device=self.device)

        wave_real = None
        wave_imag = None
        phase = None
        frequency = None

        # ✅ CORRIGIDO: Context manager válido
        if self.training:
            # Modo treinamento: permite gradientes
            for i, neuron in enumerate(self.neurons):
                out = neuron(
                    x[:, i, :, :],  # [batch, dendrites, synapses]
                    reward=reward,
                    mode=mode,
                    **neuron_kwargs,
                )
                spikes[:, i] = out['spikes']
                rates[i] = out['spike_rate']
                thetas[i] = out['theta']
                r_hats[i] = out['r_hat']

                if 'wave_real' in out:
                    if wave_real is None:
                        wave_real = torch.zeros(batch_size, self.n_neurons, device=self.device)
                        wave_imag = torch.zeros(batch_size, self.n_neurons, device=self.device)
                        phase = torch.zeros(batch_size, self.n_neurons, device=self.device)
                        frequency = torch.zeros(self.n_neurons, device=self.device)
                    wave_real[:, i] = out['wave_real']
                    wave_imag[:, i] = out['wave_imag']
                    phase[:, i] = out['phase']
                    frequency[i] = out['frequency']
        else:
            # Modo avaliação: sem gradientes (mais rápido)
            with torch.no_grad():
                for i, neuron in enumerate(self.neurons):
                    out = neuron(
                        x[:, i, :, :],
                        reward=reward,
                        mode=mode,
                        **neuron_kwargs,
                    )
                    spikes[:, i] = out['spikes']
                    rates[i] = out['spike_rate']
                    thetas[i] = out['theta']
                    r_hats[i] = out['r_hat']

                    if 'wave_real' in out:
                        if wave_real is None:
                            wave_real = torch.zeros(batch_size, self.n_neurons, device=self.device)
                            wave_imag = torch.zeros(batch_size, self.n_neurons, device=self.device)
                            phase = torch.zeros(batch_size, self.n_neurons, device=self.device)
                            frequency = torch.zeros(self.n_neurons, device=self.device)
                        wave_real[:, i] = out['wave_real']
                        wave_imag[:, i] = out['wave_imag']
                        phase[:, i] = out['phase']
                        frequency[i] = out['frequency']

        output = {
            'spikes': spikes,
            'rates': rates,
            'thetas': thetas,
            'r_hats': r_hats,
        }

        if wave_real is not None and wave_imag is not None and phase is not None and frequency is not None:
            output.update({
                'wave_real': wave_real,
                'wave_imag': wave_imag,
                'phase': phase,
                'frequency': frequency,
            })

        return output

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepara entrada para o formato [batch, neurons, dendrites, synapses].
        
        Args:
            x: Tensor em vários formatos possíveis
        
        Returns:
            Tensor no formato [batch, neurons, dendrites, synapses]
        
        Raises:
            ValueError: Se formato for inválido
        """
        # Caso 1: [B, N, D, S] - já correto
        if x.dim() == 4:
            if x.shape[1] != self.n_neurons:
                raise ValueError(
                    f"Esperado {self.n_neurons} neurônios, "
                    f"recebido {x.shape[1]}"
                )
            if x.shape[2] != self.n_dendrites:
                raise ValueError(
                    f"Esperado {self.n_dendrites} dendritos, "
                    f"recebido {x.shape[2]}"
                )
            if x.shape[3] != self.n_synapses:
                raise ValueError(
                    f"Esperado {self.n_synapses} sinapses, "
                    f"recebido {x.shape[3]}"
                )
            return x

        # Caso 2: [B, N, D] - expande sinapses
        if x.dim() == 3 and x.shape[1] == self.n_neurons:
            if x.shape[2] != self.n_dendrites:
                raise ValueError(
                    f"Esperado {self.n_dendrites} dendritos, "
                    f"recebido {x.shape[2]}"
                )
            return x.unsqueeze(-1).expand(-1, -1, -1, self.n_synapses)

        # Caso 3: [B, D, S] - mesmo para todos neurônios
        if x.dim() == 3:
            if x.shape[1] != self.n_dendrites:
                raise ValueError(
                    f"Esperado {self.n_dendrites} dendritos, "
                    f"recebido {x.shape[1]}"
                )
            if x.shape[2] != self.n_synapses:
                raise ValueError(
                    f"Esperado {self.n_synapses} sinapses, "
                    f"recebido {x.shape[2]}"
                )
            # [B, D, S] -> [B, 1, D, S] -> [B, N, D, S]
            return x.unsqueeze(1).expand(-1, self.n_neurons, -1, -1)

        # Formato inválido
        raise ValueError(
            f"Formato de entrada não suportado: {x.shape}. "
            f"Esperado: [B,{self.n_neurons},{self.n_dendrites},{self.n_synapses}] "
            f"ou [B,{self.n_neurons},{self.n_dendrites}] "
            f"ou [B,{self.n_dendrites},{self.n_synapses}]"
        )

    def set_mode(self, mode: LearningMode) -> None:
        """Define modo de aprendizado para todos os neurônios."""
        for neuron in self.neurons:
            neuron.set_mode(mode)

    def apply_batch_update(self, reward: Optional[float] = None) -> None:
        """
        Aplica plasticidade acumulada em todos os neurônios.
        Útil no modo BATCH após processar múltiplos exemplos.
        """
        for neuron in self.neurons:
            neuron.apply_plasticity(reward=reward)

    def sleep(self, duration: float = 60.0) -> None:
        """
        Ciclo de sono para consolidação two-factor em todos neurônios.
        
        Args:
            duration: Duração do sono em unidades de tempo
        """
        for neuron in self.neurons:
            neuron.sleep(duration)

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """
        Retorna métricas de todos os neurônios na camada.
        
        Returns:
            Lista de dicionários com métricas de cada neurônio
        """
        return [neuron.get_metrics() for neuron in self.neurons]

    def get_layer_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas agregadas da camada.
        
        Returns:
            Dicionário com estatísticas da camada
        """
        all_metrics = self.get_all_metrics()
        
        if not all_metrics:
            return {
                'n_neurons': self.n_neurons,
                'theta_mean': 0.0,
                'r_hat_mean': 0.0,
                'n_mean': 0.0,
                'i_mean': 0.0,
                'saturation_ratio': 0.0,
                'protection_ratio': 0.0,
            }
        
        # Agrega métricas
        return {
            'n_neurons': self.n_neurons,
            'theta_mean': sum(m['theta'] for m in all_metrics) / len(all_metrics),
            'r_hat_mean': sum(m['r_hat'] for m in all_metrics) / len(all_metrics),
            'n_mean': sum(m['N_mean'] for m in all_metrics) / len(all_metrics),
            'i_mean': sum(m['I_mean'] for m in all_metrics) / len(all_metrics),
            'saturation_ratio': sum(m['saturation_ratio'] for m in all_metrics) / len(all_metrics),
            'protection_ratio': sum(m['protection_ratio'] for m in all_metrics) / len(all_metrics),
            'total_synapses': sum(m['total_synapses'] for m in all_metrics),
        }

    def extra_repr(self) -> str:
        """Representação string da camada."""
        return (f"{self.name}: {self.n_neurons} neurons, "
                f"{self.n_dendrites}D×{self.n_synapses}S, "
                f"θμ={self.theta_mean:.2f}, r̂μ={self.r_hat_mean:.3f}")
