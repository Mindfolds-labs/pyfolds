"""Thread-safe circular buffer for telemetry."""

from typing import List, Optional, TypeVar, Generic
from threading import Lock
from collections.abc import Iterable

T = TypeVar('T')


class RingBuffer(Generic[T]):
    """
    Thread-safe circular buffer with fixed capacity.
    
    This buffer maintains items in FIFO order and overwrites oldest
    items when capacity is reached. All operations are thread-safe.
    
    Args:
        capacity: Maximum number of elements
        
    Example:
        >>> buf = RingBuffer[int](3)
        >>> buf.append(1)
        >>> buf.append(2)
        >>> buf.append(3)
        >>> buf.append(4)  # Overwrites oldest (1)
        >>> buf.snapshot()  # [2, 3, 4]
        >>> len(buf)  # 3
    """
    
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._cap = capacity
        self._buf: List[Optional[T]] = [None] * capacity
        self._idx = 0
        self._size = 0
        self._lock = Lock()
    
    def append(self, item: T) -> None:
        """Add an item to the buffer (thread-safe)."""
        with self._lock:
            if self._size < self._cap:
                self._size += 1
            self._buf[self._idx] = item
            self._idx = (self._idx + 1) % self._cap
    
    def extend(self, items: Iterable[T]) -> None:
        """Add multiple items to the buffer (thread-safe)."""
        for item in items:
            self.append(item)
    
    def snapshot(self) -> List[T]:
        """
        Return a copy of elements in chronological order (thread-safe).
        
        Returns:
            List of elements from oldest to newest
        """
        with self._lock:
            if self._size == 0:
                return []
            
            start = (self._idx - self._size) % self._cap
            result = []
            
            for i in range(self._size):
                idx = (start + i) % self._cap
                # ✅ Always append, even if None (None shouldn't happen)
                result.append(self._buf[idx])
            
            return result
    
    def clear(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self._lock:
            self._buf = [None] * self._cap
            self._idx = 0
            self._size = 0
    
    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._cap
    
    @property
    def is_full(self) -> bool:
        """Is buffer full?"""
        with self._lock:
            return self._size == self._cap
    
    def __len__(self) -> int:
        """Current number of elements."""
        with self._lock:
            return self._size
    
    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._cap}, size={self._size})"