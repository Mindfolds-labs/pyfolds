# pyfolds/advanced/inhibition.py
"""
Inibição GABA para PyFolds (Layer-level)

Implementa:
- Inibição lateral (gaussiana)
- Inibição feedforward (E→I)
- Inibição feedback (I→E)
- Esparsidade controlada

Baseado em:
    Isaacson, J. S., & Scanziani, M. (2011). How inhibition shapes cortical activity.
    Neuron, 72(2), 231-243.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict


_LOGGER = logging.getLogger(__name__)


class InhibitionLayer(nn.Module):
    """
    Camada de inibição GABA para redes MPJRD.
    
    Características:
    - Inibição lateral (gaussiana): neurônios próximos se inibem
    - Inibição feedforward (E→I): excitatórios ativam inibitórios
    - Inibição feedback (I→E): inibitórios inibem excitatórios
    - Esparsidade controlada (5-15%)
    - Overhead: ~5%
    
    Uso típico:
        exc_layer = MPJRDLayer(100, cfg)  # Excitatórios
        inh_layer = InhibitionLayer(
            n_excitatory=100,
            n_inhibitory=25,
            lateral_strength=0.5,
            feedback_strength=0.4
        )
        
        # Forward
        exc_out = exc_layer(x)
        inh_out = inh_layer(exc_out['spikes'])
        exc_out_final = inh_layer.apply_inhibition(exc_out, inh_out)
    """
    
    def __init__(
        self,
        n_excitatory: int,
        n_inhibitory: Optional[int] = None,
        lateral_strength: float = 0.5,
        feedforward_strength: float = 0.3,
        feedback_strength: float = 0.4,
        lateral_sigma: float = 5.0,
        trainable_i2e: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_excitatory: Número de neurônios excitatórios
            n_inhibitory: Número de neurônios inibitórios (padrão: n_exc//4)
            lateral_strength: Força da inibição lateral (0.0-1.0)
            feedforward_strength: Força da inibição feedforward (0.0-1.0)
            feedback_strength: Força da inibição feedback (0.0-1.0)
            lateral_sigma: Largura do kernel gaussiano para inibição lateral
            trainable_i2e: Se True, pesos I→E viram nn.Parameter treinável
            seed: Seed opcional para inicialização de conectividade
        """
        super().__init__()
        
        # Parâmetros
        self.n_exc = n_excitatory
        self.n_inh = n_inhibitory or max(1, n_excitatory // 4)
        self.lateral_strength = lateral_strength
        self.feedforward_strength = feedforward_strength
        self.feedback_strength = feedback_strength
        self.lateral_sigma = lateral_sigma
        self.seed = torch.initial_seed() if seed is None else int(seed)
        
        # ===== PESOS E→I (Feedforward) =====
        # Matriz [n_exc, n_inh] - esparsa
        self.register_buffer("W_E2I", self._init_E2I_weights())
        
        # ===== PESOS I→E (Feedback) =====
        # Matriz [n_inh, n_exc] - broadcast com decay espacial
        i2e_weights = self._init_I2E_weights()
        if trainable_i2e:
            self.W_I2E = nn.Parameter(i2e_weights)
        else:
            self.register_buffer("W_I2E", i2e_weights)
        
        # ===== KERNEL GAUSSIANO (Lateral) =====
        self.register_buffer("lateral_kernel", self._create_lateral_kernel())
        
        # ===== ESTADO DOS NEURÔNIOS INIBITÓRIOS =====
        self.register_buffer("inh_potential", torch.zeros(self.n_inh))
        self.register_buffer("inh_threshold", torch.ones(self.n_inh) * 2.0)
        self.register_buffer("step_count", torch.tensor(0))
    
    def _init_E2I_weights(self) -> torch.Tensor:
        """Inicializa pesos E→I com conectividade esparsa determinística."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        dense_weights = torch.nn.init.xavier_uniform_(
            torch.empty(self.n_exc, self.n_inh),
            generator=generator,
        )

        sparsity_level = 0.1
        mask = torch.bernoulli(
            torch.full((self.n_exc, self.n_inh), sparsity_level),
            generator=generator,
        )
        W = (dense_weights * mask).clamp(0, 1)
        return W
    
    def _init_I2E_weights(self) -> torch.Tensor:
        """
        Inicializa pesos I→E com decay espacial.
        
        Cada inibitório inibe todos os excitatórios,
        mas com força decaindo com a distância.
        """
        W = torch.zeros(self.n_inh, self.n_exc)
        
        # Assumindo neurônios organizados linearmente por índice
        for i in range(self.n_inh):
            # Centro da influência deste inibitório
            center = (i / self.n_inh) * self.n_exc
            positions = torch.arange(self.n_exc, dtype=torch.float32)
            distances = torch.abs(positions - center)
            
            # Gaussiana: força decai com a distância
            # sigma proporcional ao tamanho da população
            sigma = self.n_exc / 10.0
            W[i] = torch.exp(-distances**2 / (2 * sigma**2)) * 0.8
        
        return W
    
    def _create_lateral_kernel(self) -> torch.Tensor:
        """
        Cria kernel gaussiano para inibição lateral entre excitatórios.
        
        Returns:
            Tensor [n_exc, n_exc] com pesos de inibição lateral
        """
        positions = torch.arange(self.n_exc, dtype=torch.float32)
        kernel = torch.zeros(self.n_exc, self.n_exc)
        
        for i in range(self.n_exc):
            distances = torch.abs(positions - i)
            kernel[i] = torch.exp(-distances**2 / (2 * self.lateral_sigma**2))
        
        # Remove auto-conexão (neurônio não inibe a si mesmo)
        kernel.fill_diagonal_(0)
        
        return kernel
    
    def forward(self, exc_spikes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processa neurônios inibitórios baseado nos spikes excitatórios.
        
        Args:
            exc_spikes: [batch, n_exc] spikes dos neurônios excitatórios
            
        Returns:
            Dict com:
                'inh_spikes': [n_inh] spikes inibitórios (média sobre batch)
                'inh_potential': [n_inh] potencial dos inibitórios
                'feedforward_input': [n_inh] input feedforward
        """
        self.step_count.add_(1)

        # ===== FEEDFORWARD: E → I =====
        # [B, n_exc] @ [n_exc, n_inh] → [B, n_inh]
        feedforward_input = torch.matmul(exc_spikes, self.W_E2I.to(exc_spikes.device))
        feedforward_input *= self.feedforward_strength
        
        # ===== SPIKE INIBITÓRIO =====
        # Média sobre batch para estabilidade
        inh_input_mean = feedforward_input.mean(dim=0)  # [n_inh]
        self.inh_potential = inh_input_mean
        
        # Disparo se potencial >= threshold
        inh_spikes = (self.inh_potential >= self.inh_threshold).float()

        inh_rate = inh_spikes.float().mean().item()
        if inh_rate > 0.5:
            _LOGGER.warning(f"⚠️ Alto nível de inibição: {inh_rate:.1%}")

        return {
            'inh_spikes': inh_spikes,  # [n_inh]
            'inh_potential': self.inh_potential.clone(),
            'feedforward_input': feedforward_input.mean(dim=0),
        }
    
    def apply_inhibition(
        self,
        exc_output: Dict[str, torch.Tensor],
        inh_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Aplica inibição aos neurônios excitatórios.
        
        Args:
            exc_output: Output da camada excitatória (deve conter 'spikes' e 'u')
            inh_output: Output desta camada (de 'forward')
            
        Returns:
            exc_output modificado com inibição aplicada
        """
        device = exc_output['spikes'].device
        batch_size = exc_output['spikes'].shape[0]
        
        exc_spikes = exc_output['spikes']  # [B, n_exc]
        if exc_spikes.dim() != 2 or exc_spikes.shape[1] != self.n_exc:
            raise ValueError(
                f"Esperado exc_output['spikes'] com shape [B, {self.n_exc}], "
                f"recebido {tuple(exc_spikes.shape)}"
            )

        inh_spikes = inh_output['inh_spikes']  # [n_inh]
        if inh_spikes.dim() != 1 or inh_spikes.shape[0] != self.n_inh:
            raise ValueError(
                f"Esperado inh_output['inh_spikes'] com shape [{self.n_inh}], "
                f"recebido {tuple(inh_spikes.shape)}"
            )

        # ===== INIBIÇÃO LATERAL =====
        # [B, n_exc] @ [n_exc, n_exc] → [B, n_exc]
        lateral_kernel = self.lateral_kernel.to(device)
        lateral_inh = torch.matmul(exc_spikes, lateral_kernel)
        lateral_inh *= self.lateral_strength
        
        # ===== INIBIÇÃO FEEDBACK: I → E =====
        # [n_inh] @ [n_inh, n_exc] → [n_exc]
        W_I2E = self.W_I2E.to(device)
        feedback_inh = torch.matmul(inh_spikes, W_I2E)
        feedback_inh *= self.feedback_strength
        
        # Broadcast para batch: [1, n_exc] → [B, n_exc]
        feedback_inh = feedback_inh.unsqueeze(0).expand(batch_size, -1)
        
        # ===== INIBIÇÃO TOTAL =====
        total_inh = lateral_inh + feedback_inh
        
        # ===== RECOMPUTA SPIKES =====
        # Subtrai inibição do potencial
        u = exc_output.get('u_values', exc_output.get('u'))
        if u is None:
            raise ValueError(
                "Campo 'u_values' (ou compatível 'u') não encontrado em exc_output. "
                f"Campos disponíveis: {list(exc_output.keys())}"
            )
        if u.dim() == 3:
            # Compatível com representações [B, N, D]: agrega dimensão dendrítica.
            u = u.mean(dim=2)
        if u.dim() != 2 or u.shape[1] != self.n_exc:
            raise ValueError(
                f"Shape de potencial incompatível: {tuple(u.shape)}; esperado [B, {self.n_exc}]"
            )
        u_inhibited = u - total_inh
        
        # Recomputa spikes com threshold
        theta = exc_output.get('theta')
        if theta is None and 'thetas' in exc_output:
            theta = exc_output['thetas']

        if theta is None:
            raise ValueError(
                "Campo 'theta' (ou 'thetas') não encontrado em exc_output. "
                f"Campos disponíveis: {list(exc_output.keys())}"
            )

        if theta.dim() == 0:
            theta_expanded = torch.full_like(u_inhibited, theta.item())
        elif theta.dim() == 1:
            if theta.shape[0] == batch_size and u_inhibited.shape[1] == 1:
                theta_expanded = theta.unsqueeze(1)
            elif theta.shape[0] == u_inhibited.shape[1]:
                theta_expanded = theta.unsqueeze(0).expand(batch_size, -1)
            elif theta.shape[0] == 1:
                theta_expanded = theta.view(1, 1).expand_as(u_inhibited)
            else:
                raise ValueError(
                    f"Shape de theta incompatível: {tuple(theta.shape)} para "
                    f"u_inhibited {tuple(u_inhibited.shape)}"
                )
        elif theta.dim() == 2 and theta.shape == u_inhibited.shape:
            theta_expanded = theta
        else:
            raise ValueError(
                f"Shape de theta incompatível: {tuple(theta.shape)} para "
                f"u_inhibited {tuple(u_inhibited.shape)}"
            )

        spikes_final = (u_inhibited >= theta_expanded).float()
        
        # ===== MODIFICA OUTPUT =====
        exc_output['spikes'] = spikes_final
        exc_output['u_inhibited'] = u_inhibited
        exc_output['lateral_inh'] = lateral_inh
        exc_output['feedback_inh'] = feedback_inh
        exc_output['total_inh'] = total_inh
        exc_output['sparsity'] = spikes_final.mean()
        
        return exc_output
    
    def get_inhibition_metrics(self) -> dict:
        """
        Retorna métricas da inibição.
        
        Returns:
            dict: Dicionário com métricas
        """
        return {
            'n_excitatory': self.n_exc,
            'n_inhibitory': self.n_inh,
            'lateral_strength': self.lateral_strength,
            'feedforward_strength': self.feedforward_strength,
            'feedback_strength': self.feedback_strength,
            'lateral_sigma': self.lateral_sigma,
            'step': self.step_count.item(),
            'inh_potential_mean': self.inh_potential.mean().item(),
            'inh_threshold_mean': self.inh_threshold.mean().item(),
        }
    
    def extra_repr(self) -> str:
        return (f"E={self.n_exc}, I={self.n_inh}, "
                f"lateral={self.lateral_strength:.2f}, "
                f"feedback={self.feedback_strength:.2f}")


class InhibitionMixin:
    """
    Mixin para integrar inibição em layers.
    
    NOTA: Este mixin opera no nível de Layer, não de neurônio individual.
    Deve ser usado com MPJRDLayer ou classes derivadas.
    
    Uso:
        class InhibitedLayer(InhibitionMixin, MPJRDLayer):
            def __init__(self, n_neurons, cfg):
                super().__init__(n_neurons, cfg)
                self._init_inhibition()
    """
    
    def _init_inhibition(
        self,
        n_inhibitory: Optional[int] = None,
        lateral_strength: float = 0.5,
        feedforward_strength: float = 0.3,
        feedback_strength: float = 0.4,
        lateral_sigma: float = 5.0,
        trainable_i2e: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        """
        Inicializa camada inibitória.
        
        Args:
            n_inhibitory: Número de neurônios inibitórios (padrão: n_neurons//4)
            lateral_strength: Força da inibição lateral
            feedforward_strength: Força da inibição feedforward
            feedback_strength: Força da inibição feedback
            lateral_sigma: Largura do kernel gaussiano
            trainable_i2e: Define se I→E é treinável (default usa cfg)
            seed: Seed opcional para conectividade
        """
        cfg = getattr(self, "cfg", None)
        if trainable_i2e is None:
            trainable_i2e = bool(getattr(cfg, "inhibition_trainable_i2e", False))
        if seed is None and cfg is not None:
            seed = getattr(cfg, "random_seed", None)

        self.inhibition = InhibitionLayer(
            n_excitatory=self.n_neurons,
            n_inhibitory=n_inhibitory,
            lateral_strength=lateral_strength,
            feedforward_strength=feedforward_strength,
            feedback_strength=feedback_strength,
            lateral_sigma=lateral_sigma,
            trainable_i2e=trainable_i2e,
            seed=seed,
        )
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward com inibição aplicada.
        
        Args:
            x: Tensor de entrada [batch, n_neurons, dendrites, synapses]
            **kwargs: Argumentos adicionais para a camada base
        
        Returns:
            Dict com spikes após inibição
        """
        if not hasattr(self, 'inhibition'):
            raise RuntimeError(
                "Inibição não foi inicializada. Chame _init_inhibition() antes de forward()."
            )

        # Forward excitatório normal
        exc_output = super().forward(x, **kwargs)

        # Processa inibitórios
        inh_output = self.inhibition(exc_output['spikes'])
        
        # Aplica inibição
        final_output = self.inhibition.apply_inhibition(exc_output, inh_output)
        
        return final_output
    
    def get_inhibition_metrics(self) -> dict:
        """Retorna métricas da inibição desta layer."""
        if hasattr(self, 'inhibition'):
            return self.inhibition.get_inhibition_metrics()
        return {}
