"""Sinks (destinos) para eventos de telemetria"""

import json
import warnings
from typing import List, Optional
from .events import TelemetryEvent
from .ringbuffer import RingBuffer

# Tentativa de importar torch para detecção de tensores
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Sink:
    """Classe base para sinks de telemetria."""
    
    def emit(self, event: TelemetryEvent) -> None:
        """Emite um evento para este sink."""
        raise NotImplementedError
    
    def flush(self) -> None:
        """Força escrita de buffers (se aplicável)."""
        pass
    
    def close(self) -> None:
        """Fecha o sink (libera recursos)."""
        pass


class NoOpSink(Sink):
    """Sink que não faz nada (para profile=off)."""
    
    def emit(self, event: TelemetryEvent) -> None:
        pass


class MemorySink(Sink):
    """
    Sink que mantém eventos em memória (buffer circular).
    
    Args:
        capacity: Capacidade máxima do buffer
    """
    
    def __init__(self, capacity: int = 512):
        self.buffer = RingBuffer[TelemetryEvent](capacity)
    
    def emit(self, event: TelemetryEvent) -> None:
        self.buffer.append(event)
    
    def snapshot(self) -> List[TelemetryEvent]:
        """Retorna cópia dos eventos no buffer."""
        return self.buffer.snapshot()
    
    def clear(self) -> None:
        """Limpa o buffer."""
        # ✅ CORRIGIDO: Usa property pública capacity
        self.buffer = RingBuffer[TelemetryEvent](self.buffer.capacity)


class ConsoleSink(Sink):
    """
    Sink que imprime eventos no console.
    
    Args:
        verbose: Se True, imprime payload completo
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def emit(self, event: TelemetryEvent) -> None:
        if self.verbose:
            print(f"[pyfolds] step={event.step_id} "
                  f"phase={event.phase} "
                  f"mode={event.mode} "
                  f"neuron={event.neuron_id} "
                  f"payload={event.payload}")
        else:
            print(f"[pyfolds] step={event.step_id} phase={event.phase}")


class JSONLinesSink(Sink):
    """
    Sink que escreve eventos em arquivo JSON Lines.
    
    Args:
        path: Caminho do arquivo
        flush_every: Número de eventos para flush automático (0 = nunca)
    
    Example:
        >>> with JSONLinesSink("telemetry.jsonl") as sink:
        ...     sink.emit(event)
        ... # Arquivo fechado automaticamente
    """
    
    def __init__(self, path: str, flush_every: int = 10):
        self.path = path
        self.flush_every = flush_every
        self._count = 0
        self._file = None
    
    def __enter__(self):
        """Abre arquivo ao entrar no context manager."""
        self._file = open(self.path, 'a', encoding='utf-8')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Garante fechamento mesmo com erro."""
        self.close()
    
    def _ensure_open(self):
        """Abre arquivo se necessário (para uso sem context manager)."""
        if self._file is None:
            self._file = open(self.path, 'a', encoding='utf-8')
    
    def _make_serializable(self, obj):
        """Converte objetos não serializáveis para formatos JSON compatíveis."""
        if TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.detach().cpu().tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Fallback para string
            return str(obj)
    
    def emit(self, event: TelemetryEvent) -> None:
        """Emite evento para o arquivo."""
        self._ensure_open()
        
        try:
            data = {
                'step_id': event.step_id,
                'phase': event.phase,
                'mode': event.mode,
                'neuron_id': event.neuron_id,
                'timestamp': event.timestamp,
                'wall_time': event.wall_time,
                'payload': event.payload
            }
            self._file.write(json.dumps(data) + '\n')
            
        except (TypeError, ValueError) as e:
            # Se falhar na serialização, tenta converter objetos não serializáveis
            warnings.warn(f"Falha ao serializar evento: {e}. Tentando converter...")
            
            # Converte payload para formato serializável
            serializable_data = {
                'step_id': event.step_id,
                'phase': event.phase,
                'mode': event.mode,
                'neuron_id': event.neuron_id,
                'timestamp': event.timestamp,
                'wall_time': event.wall_time,
                'payload': self._make_serializable(event.payload)
            }
            self._file.write(json.dumps(serializable_data) + '\n')
        
        self._count += 1
        if self.flush_every > 0 and self._count % self.flush_every == 0:
            self.flush()
    
    def flush(self) -> None:
        """Força escrita no disco."""
        if self._file is not None:
            self._file.flush()
    
    def close(self) -> None:
        """Fecha o arquivo."""
        if self._file is not None:
            self._file.close()
            self._file = None


# ===== DistributorSink =====
class DistributorSink(Sink):
    """
    Sink que distribui eventos para múltiplos sinks.
    
    Útil para enviar eventos para diferentes destinos simultaneamente:
    - MemorySink para MindBoard (tempo real)
    - JSONLinesSink para MindAudit (persistência)
    - ConsoleSink para debug
    
    Args:
        sinks: Lista de sinks para onde os eventos serão distribuídos
    
    Example:
        >>> sink = DistributorSink([
        ...     MemorySink(1000),           # Para MindBoard
        ...     JSONLinesSink("audit.jsonl") # Para MindAudit
        ... ])
        >>> telem = TelemetryController(sink=sink)
    """
    
    def __init__(self, sinks: List[Sink]):
        self.sinks = sinks
    
    def emit(self, event: TelemetryEvent) -> None:
        """Distribui evento para todos os sinks."""
        for sink in self.sinks:
            sink.emit(event)
    
    def flush(self) -> None:
        """Flush em todos os sinks que suportam."""
        for sink in self.sinks:
            if hasattr(sink, 'flush'):
                sink.flush()
    
    def close(self) -> None:
        """Fecha todos os sinks."""
        for sink in self.sinks:
            if hasattr(sink, 'close'):
                sink.close()