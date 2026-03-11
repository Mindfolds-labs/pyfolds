"""Mixin base para gestão compartilhada de tempo."""

import torch
import torch.nn as nn


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
    
    def _validate_module_instance(self) -> nn.Module:
        """Garante contrato mínimo para mixins que registram buffers."""
        if not isinstance(self, nn.Module):
            raise TypeError(
                f"{self.__class__.__name__} deve herdar de torch.nn.Module para usar TimedMixin"
            )
        return self

    def _ensure_time_counter(self):
        """
        Garante que time_counter existe.
        
        Chamado por mixins que precisam de contador de tempo para garantir
        que apenas UM contador seja criado, evitando conflitos.
        """
        module = self._validate_module_instance()
        if not hasattr(module, 'time_counter'):
            module.register_buffer("time_counter", torch.tensor(0.0))
    
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
