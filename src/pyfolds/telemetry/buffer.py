"""Thread-safe fixed-size ring buffer for telemetry events."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


class RingBufferThreadSafe(Generic[T]):
    """Thread-safe ring buffer using a small critical section around deque operations."""

    def __init__(self, capacity: int, drop_oldest_on_overflow: bool = True):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.drop_oldest_on_overflow = drop_oldest_on_overflow
        self._dq: deque[T] = deque(maxlen=capacity if drop_oldest_on_overflow else None)
        self._lock = Lock()
        self._dropped = 0

    def push(self, event: T) -> bool:
        with self._lock:
            if len(self._dq) >= self.capacity and not self.drop_oldest_on_overflow:
                self._dropped += 1
                return False
            if len(self._dq) >= self.capacity and self.drop_oldest_on_overflow:
                self._dropped += 1
            self._dq.append(event)
            return True

    def snapshot(self, max_events: int | None = None) -> list[T]:
        with self._lock:
            items = list(self._dq)
        if max_events is None:
            return items
        return items[-max_events:]

    def drain(self, max_events: int | None = None) -> list[T]:
        with self._lock:
            if max_events is None or max_events >= len(self._dq):
                items = list(self._dq)
                self._dq.clear()
                return items
            items = [self._dq.popleft() for _ in range(max_events)]
            return items

    def clear(self) -> None:
        with self._lock:
            self._dq.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._dq)

    def dropped_events_count(self) -> int:
        with self._lock:
            return self._dropped
