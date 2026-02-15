"""Buffer circular thread-safe para telemetria"""

from typing import List, Optional, TypeVar, Generic
from threading import Lock
from collections.abc import Iterable

T = TypeVar('T')

class RingBuffer(Generic[T]):
    """
    Buffer circular thread-safe com capacidade fixa.
    
    Args:
        capacity: Número máximo de elementos
        
    Example:
        >>> buf = RingBuffer[int](3)
        >>> buf.append(1)
        >>> buf.append(2)
        >>> buf.append(3)
        >>> buf.append(4)  # Sobrescreve o mais antigo
        >>> buf.snapshot()  # [2, 3, 4]
        >>> len(buf)  # 3
    """
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self._cap = capacity
        self._buf: List[Optional[T]] = [None] * capacity
        self._idx = 0
        self._size = 0
        self._lock = Lock()
    
    def append(self, item: T) -> None:
        """Adiciona um item ao buffer."""
        with self._lock:
            if self._size < self._cap:
                self._size += 1
            self._buf[self._idx] = item
            self._idx = (self._idx + 1) % self._cap
    
    def extend(self, items: Iterable[T]) -> None:
        """Adiciona múltiplos itens ao buffer."""
        for item in items:
            self.append(item)
    
    def snapshot(self) -> List[T]:
        """
        Retorna uma cópia dos elementos em ordem cronológica.
        
        Returns:
            Lista com os elementos do mais antigo ao mais novo
        """
        with self._lock:
            if self._size == 0:
                return []
            
            start = (self._idx - self._size) % self._cap
            result = []
            
            for i in range(self._size):
                idx = (start + i) % self._cap
                val = self._buf[idx]
                if val is not None:
                    result.append(val)
            
            return result
    
    def clear(self) -> None:
        """Limpa o buffer."""
        with self._lock:
            self._buf = [None] * self._cap
            self._idx = 0
            self._size = 0
    
    @property
    def capacity(self) -> int:
        """Capacidade máxima do buffer."""
        return self._cap
    
    @property
    def is_full(self) -> bool:
        """Buffer está cheio?"""
        return self._size == self._cap
    
    def __len__(self) -> int:
        """Número atual de elementos."""
        return self._size
    
    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._cap}, size={self._size})"