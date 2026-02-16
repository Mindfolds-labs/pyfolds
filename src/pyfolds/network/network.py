"""Rede MPJRD completa - VERSÃO CORRIGIDA E OTIMIZADA"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union, Tuple
from collections import defaultdict, deque
from ..layers import MPJRDLayer
from ..utils.types import LearningMode


class MPJRDNetwork(nn.Module):
    """
    Rede neural completa com camadas MPJRD.
    
    Características:
        - Múltiplas camadas de neurônios MPJRD
        - Conexões configuráveis entre camadas
        - Ordenação topológica automática (algoritmo de Kahn)
        - Detecção de ciclos
        - Preparação inteligente de entrada para camadas
        - Modos de aprendizado por camada
        - Métricas agregadas da rede
    
    Args:
        name: Nome da rede (opcional)
    
    Example:
        >>> from pyfolds import MPJRDConfig, MPJRDLayer
        >>> cfg = MPJRDConfig(n_dendrites=4)
        >>> 
        >>> net = MPJRDNetwork("minha_rede")
        >>> net.add_layer("input", MPJRDLayer(10, cfg))
        >>> net.add_layer("hidden", MPJRDLayer(20, cfg))
        >>> net.add_layer("output", MPJRDLayer(5, cfg))
        >>> net.connect("input", "hidden")
        >>> net.connect("hidden", "output")
        >>> net.build()
        >>> 
        >>> x = torch.randn(32, 10, 4, 32)  # [batch, neurons, dendrites, synapses]
        >>> out = net(x)
        >>> print(out['output'].shape)  # [32, 5]
    """

    def __init__(self, name: str = "MPJRDNetwork"):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleDict()
        self.connections = []  # Lista de tuplas (origem, destino)
        self.connection_weights = nn.ParameterDict()  # Pesos das conexões
        self.built = False
        self.input_layer = None
        self.output_layer = None
        self._layer_order = []  # Cache da ordenação topológica

    def add_layer(self, name: str, layer: MPJRDLayer) -> 'MPJRDNetwork':
        """
        Adiciona uma camada à rede.
        
        Args:
            name: Nome único da camada
            layer: Camada MPJRDLayer
        
        Returns:
            self (para encadeamento)
        """
        if name in self.layers:
            raise ValueError(f"Camada '{name}' já existe na rede")
        
        self.layers[name] = layer
        
        # Se for a primeira camada, define como entrada
        if len(self.layers) == 1:
            self.input_layer = name
        
        return self

    def connect(self, from_layer: str, to_layer: str, 
                weight_init: float = 1.0) -> 'MPJRDNetwork':
        """
        Conecta duas camadas com validação de ciclos.
        
        Args:
            from_layer: Nome da camada de origem
            to_layer: Nome da camada de destino
            weight_init: Inicialização dos pesos (padrão: 1.0)
        
        Returns:
            self (para encadeamento)
        """
        if from_layer not in self.layers:
            raise ValueError(f"Camada origem '{from_layer}' não encontrada")
        
        if to_layer not in self.layers:
            raise ValueError(f"Camada destino '{to_layer}' não encontrada")
        
        # Verifica se já existe conexão
        conn = (from_layer, to_layer)
        if conn in self.connections:
            raise ValueError(f"Conexão {from_layer}→{to_layer} já existe")
        
        # ✅ VALIDAÇÃO DE CICLO
        test_connections = self.connections + [conn]
        if self._has_cycle(test_connections):
            raise ValueError(
                f"Conexão {from_layer}→{to_layer} criaria um ciclo! "
                f"Redes feedforward não permitem ciclos."
            )
        
        self.connections.append(conn)
        
        # Cria pesos para esta conexão
        conn_name = f"{from_layer}_to_{to_layer}"
        from_neurons = self.layers[from_layer].n_neurons
        to_neurons = self.layers[to_layer].n_neurons
        
        self.connection_weights[conn_name] = nn.Parameter(
            torch.full((from_neurons, to_neurons), weight_init)
        )
        
        return self

    def _has_cycle(self, connections: List[Tuple[str, str]]) -> bool:
        """
        Detecta ciclos no grafo de conexões usando DFS.
        
        Args:
            connections: Lista de tuplas (origem, destino)
            
        Returns:
            True se houver ciclo, False caso contrário
        """
        graph = defaultdict(list)
        for from_l, to_l in connections:
            graph[from_l].append(to_l)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True  # Ciclo detectado!
            
            rec_stack.remove(node)
            return False
        
        for layer in self.layers:
            if layer not in visited:
                if dfs(layer):
                    return True
        
        return False

    def _topological_sort(self) -> List[str]:
        """
        Ordenação topológica usando algoritmo de Kahn.
        
        Returns:
            Lista de nomes de camadas em ordem topológica
            
        Raises:
            ValueError: Se houver ciclo no grafo
        """
        # Constrói grafo
        graph = defaultdict(list)
        in_degree = {layer: 0 for layer in self.layers}
        
        for from_l, to_l in self.connections:
            graph[from_l].append(to_l)
            in_degree[to_l] += 1
        
        # Kahn's algorithm
        queue = deque([l for l in self.layers if in_degree[l] == 0])
        result = []
        
        while queue:
            layer = queue.popleft()
            result.append(layer)
            
            for neighbor in graph[layer]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Verifica se há ciclo
        if len(result) != len(self.layers):
            raise ValueError(
                "Ciclo detectado no grafo de camadas! "
                "Redes feedforward não permitem ciclos."
            )
        
        return result

    def _prepare_input_for_layer(
        self, 
        spikes: torch.Tensor,  # [B, from_neurons]
        to_layer: str,
        weights: torch.Tensor  # [from, to]
    ) -> torch.Tensor:
        """
        Prepara spikes de origem para entrada da camada destino.
        
        ✅ OTIMIZADO: Distribui ativação entre dendritos de forma realista
        
        Args:
            spikes: Spikes da camada origem [B, from]
            to_layer: Nome da camada destino
            weights: Pesos da conexão [from, to]
        
        Returns:
            Tensor [B, to, D, S] pronto para a camada destino
        """
        B = spikes.shape[0]
        to_layer_obj = self.layers[to_layer]
        D = to_layer_obj.n_dendrites
        S = to_layer_obj.n_synapses
        device = spikes.device
        
        # spikes * weights -> [B, to]
        weighted = spikes @ weights  # [B, to]
        
        # Distribui em padrão esparso e estável entre dendritos/sinapses
        input_tensor = torch.zeros(B, to_layer_obj.n_neurons, D, S, device=device)

        # Gerador controlado por seed global/configurável
        generator = torch.Generator(device='cpu')
        cfg_seed = getattr(getattr(to_layer_obj, 'cfg', None), 'random_seed', None)
        seed = torch.initial_seed() if cfg_seed is None else int(cfg_seed)
        generator.manual_seed(seed)

        ratio = getattr(
            to_layer_obj,
            'active_synapses_ratio',
            getattr(getattr(to_layer_obj, 'cfg', None), 'active_synapses_ratio', 0.25),
        )
        active_synapses = min(S, max(1, int(S * ratio)))
        for d_idx in range(D):
            syn_indices = torch.randperm(S, generator=generator)[:active_synapses]
            input_tensor[:, :, d_idx, syn_indices] = (
                weighted.unsqueeze(-1) / float(active_synapses)
            )

        return input_tensor

    def build(self) -> 'MPJRDNetwork':
        """
        Constrói a rede (valida conexões, ordena topologicamente).
        
        Returns:
            self (para encadeamento)
        """
        if len(self.layers) == 0:
            raise ValueError("Rede não possui camadas")
        
        # Valida se todas as camadas estão conectadas
        if len(self.connections) == 0 and len(self.layers) > 1:
            raise ValueError("Múltiplas camadas sem conexões")
        
        # ✅ ORDENAÇÃO TOPOLÓGICA
        self._layer_order = self._topological_sort()
        
        # Define camada de entrada (primeira na ordem)
        self.input_layer = self._layer_order[0]
        
        # Define camada de saída (última na ordem)
        self.output_layer = self._layer_order[-1]
        
        self.built = True
        return self

    def forward(self, x: torch.Tensor,
                reward: Optional[float] = None,
                mode: LearningMode = LearningMode.ONLINE,
                layer_kwargs: Optional[Dict[str, Dict[str, object]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass da rede com ordenação topológica real.
        
        Args:
            x: Tensor de entrada (formato depende da primeira camada)
            reward: Sinal de recompensa (opcional)
            mode: Modo de aprendizado
        
        Returns:
            Dict com:
                - output: spikes da última camada [B, output_neurons]
                - layers: saídas de todas as camadas
                - final_layer: nome da última camada
                - layer_order: ordem de processamento
        """
        if not self.built:
            raise RuntimeError("Rede não foi construída. Chame .build() primeiro.")
        
        # ✅ USA ORDENAÇÃO TOPOLÓGICA CACHEADA
        layer_order = self._layer_order
        
        layer_kwargs = layer_kwargs or {}

        # Dicionário para armazenar saídas de cada camada
        outputs = {}
        
        # Forward da primeira camada
        first_layer = layer_order[0]
        outputs[first_layer] = self.layers[first_layer](
            x,
            reward=reward,
            mode=mode,
            **layer_kwargs.get(first_layer, {}),
        )
        
        # Processa camadas restantes em ordem
        for layer_name in layer_order[1:]:
            # Encontra todas as camadas que se conectam a esta
            inputs_to_this = []
            
            for from_layer, to_layer in self.connections:
                if to_layer == layer_name and from_layer in outputs:
                    spikes = outputs[from_layer]['spikes']  # [B, from]
                    conn_name = f"{from_layer}_to_{to_layer}"
                    weights = self.connection_weights[conn_name]  # [from, to]
                    
                    # ✅ PREPARA ENTRADA COM DISTRIBUIÇÃO DENDRÍTICA
                    prepared = self._prepare_input_for_layer(
                        spikes, layer_name, weights
                    )
                    inputs_to_this.append(prepared)
            
            # Combina múltiplas entradas (soma)
            if inputs_to_this:
                combined_input = sum(inputs_to_this)
            else:
                # Se não tem entrada, usa zeros
                B = x.shape[0]
                to_layer = self.layers[layer_name]
                combined_input = torch.zeros(
                    B, to_layer.n_neurons, to_layer.n_dendrites, to_layer.n_synapses,
                    device=x.device
                )
            
            # Forward da camada
            outputs[layer_name] = self.layers[layer_name](
                combined_input,
                reward=reward,
                mode=mode,
                **layer_kwargs.get(layer_name, {}),
            )
        
        # Resultado final
        return {
            'output': outputs[self.output_layer]['spikes'],
            'layers': outputs,
            'final_layer': self.output_layer,
            'layer_order': layer_order
        }

    def set_mode(self, mode: LearningMode) -> None:
        """Define modo de aprendizado para todas as camadas."""
        for layer in self.layers.values():
            layer.set_mode(mode)

    def apply_batch_update(self, reward: Optional[float] = None) -> None:
        """
        Aplica plasticidade acumulada em todas as camadas.
        Útil no modo BATCH após processar múltiplos exemplos.
        """
        for layer in self.layers.values():
            layer.apply_batch_update(reward=reward)

    def sleep(self, duration: float = 60.0) -> None:
        """
        Ciclo de sono para consolidação two-factor em todas camadas.
        
        Args:
            duration: Duração do sono em unidades de tempo
        """
        for layer in self.layers.values():
            layer.sleep(duration)

    def get_all_metrics(self) -> Dict[str, List[Dict]]:
        """
        Retorna métricas de todas as camadas.
        
        Returns:
            Dict com métricas por camada
        """
        return {
            name: layer.get_all_metrics()
            for name, layer in self.layers.items()
        }

    def get_network_metrics(self) -> Dict[str, float]:
        """
        Retorna métricas agregadas da rede.
        
        Returns:
            Dict com estatísticas da rede
        """
        total_neurons = sum(layer.n_neurons for layer in self.layers.values())
        total_synapses = sum(layer.n_neurons * layer.n_dendrites * layer.n_synapses 
                            for layer in self.layers.values())
        
        # Média dos thresholds por camada
        theta_means = []
        for layer in self.layers.values():
            metrics = layer.get_layer_metrics()
            theta_means.append(metrics.get('theta_mean', 0.0))
        
        return {
            'n_layers': len(self.layers),
            'n_connections': len(self.connections),
            'total_neurons': total_neurons,
            'total_synapses': total_synapses,
            'theta_mean_network': sum(theta_means) / len(theta_means) if theta_means else 0.0,
            'built': self.built
        }

    def extra_repr(self) -> str:
        return (f"{self.name}: {len(self.layers)} layers, "
                f"{len(self.connections)} connections, "
                f"built={self.built}")
