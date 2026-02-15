"""Mixin base para gestão compartilhada de tempo."""

import torch


class TimedMixin:
    """
    Mixin base para mecanismos que precisam de contador de tempo.
    
    Este mixin resolve conflitos onde múltiplos mixins (Refractory, Backprop)
    precisariam criar seus próprios contadores de tempo.
    
    Uso:
        class RefractoryMixin(TimedMixin):
            def __init__(self):
                self._ensure_time_counter()  # Garante que time_counter existe
                # ...
    
    Example:
        >>> class Neuron(RefractoryMixin, BackpropMixin, MPJRDNeuron):
        ...     def __init__(self, cfg):
        ...         self._ensure_time_counter()  # Único contador compartilhado
        ...         # ...
    """
    
    def _ensure_time_counter(self):
        """
        Garante que time_counter existe.
        
        Chamado por mixins que precisam de contador de tempo para garantir
        que apenas UM contador seja criado, evitando conflitos.
        """
        if not hasattr(self, 'time_counter'):
            self.register_buffer("time_counter", torch.tensor(0.0))
    
    def _increment_time(self, dt: float = 1.0):
        """
        Incrementa contador de tempo.
        
        Args:
            dt: Passo de tempo a ser adicionado (ms)
        """
        if hasattr(self, 'time_counter'):
            self.time_counter.add_(dt)
    
    def _get_time(self) -> float:
        """
        Retorna o tempo atual.
        
        Returns:
            float: Valor atual do contador de tempo
        """
        if hasattr(self, 'time_counter'):
            return self.time_counter.item()
        return 0.0