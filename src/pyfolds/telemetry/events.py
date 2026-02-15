"""Eventos de telemetria para neurônios MPJRD"""

import time
from dataclasses import dataclass, field
from typing import Literal, Any, Dict, Optional, Union, Callable

Phase = Literal["forward", "commit", "sleep", "mode_change"]

# Tipo para payload que pode ser eager ou lazy
PayloadType = Union[Dict[str, Any], Callable[[], Dict[str, Any]]]


@dataclass(frozen=True)
class TelemetryEvent:
    """
    Evento base de telemetria com suporte a lazy evaluation.
    
    Attributes:
        step_id: ID do passo atual
        phase: Fase do evento (forward, commit, sleep)
        mode: Modo de aprendizado
        _payload: Dicionário ou função que retorna o payload
        timestamp: Timestamp de alta precisão (perf_counter)
        wall_time: Timestamp de parede (para análise temporal)
        neuron_id: ID opcional do neurônio
    """
    step_id: int
    phase: Phase
    mode: str
    _payload: PayloadType
    timestamp: float = field(default_factory=lambda: time.perf_counter())  # ⚡ Alta precisão
    wall_time: float = field(default_factory=lambda: time.time())          # ⏰ Para referência
    neuron_id: Optional[str] = None
    
    @property
    def payload(self) -> Dict[str, Any]:
        """Avalia payload lazy se necessário."""
        if callable(self._payload):
            return self._payload()
        return self._payload


def forward_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                  **payload: Any) -> TelemetryEvent:
    """
    Cria evento de forward pass (eager).
    
    Args:
        step_id: ID do passo
        mode: Modo de aprendizado
        neuron_id: ID do neurônio (opcional)
        **payload: Métricas a serem coletadas
    """
    return TelemetryEvent(
        step_id=step_id,
        phase="forward",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


def forward_event_lazy(step_id: int, mode: str,
                       payload_fn: Callable[[], Dict[str, Any]],
                       neuron_id: Optional[str] = None) -> TelemetryEvent:
    """
    Cria evento de forward pass com lazy evaluation (economia de CPU).
    
    Args:
        step_id: ID do passo
        mode: Modo de aprendizado
        neuron_id: ID do neurônio (opcional)
        payload_fn: Função que retorna o payload (só é chamada se o evento for emitido)
    
    Example:
        >>> telem.emit(forward_event_lazy(
        ...     step_id=step,
        ...     mode="online",
        ...     payload_fn=lambda: {
        ...         'spike_rate': neuron.get_spike_rate(),  # Só calcula se necessário
        ...         'theta': neuron.theta.item()
        ...     }
        ... ))
    """
    return TelemetryEvent(
        step_id=step_id,
        phase="forward",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )


def commit_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                 **payload: Any) -> TelemetryEvent:
    """Cria evento de commit (plasticidade)."""
    return TelemetryEvent(
        step_id=step_id,
        phase="commit",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


def commit_event_lazy(step_id: int, mode: str,
                      payload_fn: Callable[[], Dict[str, Any]],
                      neuron_id: Optional[str] = None) -> TelemetryEvent:
    """Cria evento de commit com lazy evaluation."""
    return TelemetryEvent(
        step_id=step_id,
        phase="commit",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )


def sleep_event(step_id: int, mode: str, neuron_id: Optional[str] = None, 
                **payload: Any) -> TelemetryEvent:
    """Cria evento de sleep (consolidação)."""
    return TelemetryEvent(
        step_id=step_id,
        phase="sleep",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload
    )


def sleep_event_lazy(step_id: int, mode: str,
                     payload_fn: Callable[[], Dict[str, Any]],
                     neuron_id: Optional[str] = None) -> TelemetryEvent:
    """Cria evento de sleep com lazy evaluation."""
    return TelemetryEvent(
        step_id=step_id,
        phase="sleep",
        mode=mode,
        neuron_id=neuron_id,
        _payload=payload_fn
    )
