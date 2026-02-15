"""Controlador de telemetria para neurônios MPJRD"""

import random
from dataclasses import dataclass
from typing import Optional, Literal, List, Dict, Any
from .events import TelemetryEvent
from .sinks import Sink, NoOpSink, MemorySink
from .ringbuffer import RingBuffer

Profile = Literal["off", "light", "heavy"]


@dataclass
class TelemetryConfig:
    """Configuração da telemetria."""
    profile: Profile = "off"
    sample_every: int = 1
    memory_capacity: int = 512
    
    def __post_init__(self):
        """Valida configuração."""
        if self.profile == "light" and self.sample_every < 1:
            self.sample_every = 50  # Default para light
        elif self.profile == "heavy" and self.sample_every < 1:
            self.sample_every = 1   # Heavy = todos os passos


class TelemetryController:
    """
    Controlador de telemetria.
    
    Gerencia coleta, filtragem e armazenamento de eventos de telemetria.
    
    Example:
        >>> cfg = TelemetryConfig(profile="light", sample_every=50)
        >>> telem = TelemetryController(cfg)
        >>> if telem.enabled() and telem.should_emit(step_id):
        ...     telem.emit(event)
        >>> snapshot = telem.snapshot()
    """
    
    def __init__(self, cfg: Optional[TelemetryConfig] = None, sink: Optional[Sink] = None):
        """
        Args:
            cfg: Configuração da telemetria
            sink: Destino dos eventos (MemorySink, ConsoleSink, etc)
        """
        self.cfg = cfg or TelemetryConfig()
        
        # Escolhe sink baseado no perfil
        if sink is not None:
            self.sink = sink
        elif self.cfg.profile == "off":
            self.sink = NoOpSink()
        else:
            self.sink = MemorySink(self.cfg.memory_capacity)
            
        self.step_count = 0
    
    def enabled(self) -> bool:
        """Telemetria está ativada?"""
        return self.cfg.profile != "off"
    
    def should_emit(self, step_id: int) -> bool:
        """
        Deve emitir evento neste passo baseado em sample_every?
        
        Args:
            step_id: ID do passo atual
        
        Returns:
            True se deve emitir (baseado em sample_every)
        """
        if self.cfg.profile == "off":
            return False
        
        # Profile "heavy" sempre emite
        if self.cfg.profile == "heavy":
            return True
        
        # Profile "light" emite a cada sample_every passos
        every = max(1, self.cfg.sample_every)
        return (step_id % every) == 0
    
    def should_emit_sample(self, sample_rate: float) -> bool:
        """
        Amostragem probabilística independente do step_id.
        
        Args:
            sample_rate: Probabilidade de emitir (0.0 a 1.0)
        
        Returns:
            True se deve emitir baseado em amostragem aleatória
        
        Example:
            >>> # 10% das chamadas, independente do step_id
            >>> if telem.should_emit_sample(0.1):
            ...     telem.emit(event)
        """
        if self.cfg.profile == "off":
            return False
        if sample_rate >= 1.0:
            return True
        return random.random() < sample_rate
    
    def emit(self, event: TelemetryEvent) -> None:
        """
        Emite um evento de telemetria.
        
        Args:
            event: Evento a ser emitido (pode ter lazy payload)
        """
        if not self.should_emit(event.step_id):
            return
        self.sink.emit(event)
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """
        Retorna snapshot dos eventos armazenados.
        
        Returns:
            Lista de payloads dos eventos
        """
        if isinstance(self.sink, MemorySink):
            return [e.payload for e in self.sink.buffer.snapshot()]
        return []
    
    def clear(self) -> None:
        """Limpa buffer de memória."""
        if isinstance(self.sink, MemorySink):
            self.sink.buffer = RingBuffer[TelemetryEvent](self.cfg.memory_capacity)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da telemetria."""
        return {
            'profile': self.cfg.profile,
            'sample_every': self.cfg.sample_every,
            'step_count': self.step_count,
            'sink_type': type(self.sink).__name__,
            'buffer_size': len(self.sink.buffer) if hasattr(self.sink, 'buffer') else 0
        }