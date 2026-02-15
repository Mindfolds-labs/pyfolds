"""Tests for RingBuffer."""

import pytest
import threading
from pyfolds.telemetry import RingBuffer


class TestRingBuffer:
    """Test ring buffer."""
    
    def test_append(self):
        """Test FIFO order and overwrite behavior."""
        buf = RingBuffer(3)
        
        buf.append(1)
        buf.append(2)
        buf.append(3)
        assert buf.snapshot() == [1, 2, 3]
        
        buf.append(4)  # Overwrites oldest
        assert buf.snapshot() == [2, 3, 4]
        
        buf.append(5)  # Overwrites next
        assert buf.snapshot() == [3, 4, 5]
    
    def test_snapshot_with_none(self):
        """Test that snapshot includes None values (doesn't filter them out)."""
        buf = RingBuffer(3)
        buf.append(1)
        buf.append(None)
        buf.append(2)
        
        # Should include None, not filter it out
        assert buf.snapshot() == [1, None, 2]
    
    def test_empty_snapshot(self):
        """Test snapshot on empty buffer."""
        buf = RingBuffer(5)
        assert buf.snapshot() == []
    
    def test_len(self):
        """Test __len__ method."""
        buf = RingBuffer(3)
        assert len(buf) == 0
        
        buf.append(1)
        assert len(buf) == 1
        
        buf.append(2)
        buf.append(3)
        assert len(buf) == 3
        
        buf.append(4)  # Overwrite, size stays at capacity
        assert len(buf) == 3
    
    def test_clear(self):
        """Test clear method."""
        buf = RingBuffer(3)
        buf.append(1)
        buf.append(2)
        buf.append(3)
        
        buf.clear()
        assert len(buf) == 0
        assert buf.snapshot() == []
    
    def test_capacity_property(self):
        """Test capacity property."""
        buf = RingBuffer(5)
        assert buf.capacity == 5
    
    def test_is_full(self):
        """Test is_full property."""
        buf = RingBuffer(2)
        assert not buf.is_full
        
        buf.append(1)
        assert not buf.is_full
        
        buf.append(2)
        assert buf.is_full
        
        buf.append(3)  # Overwrite, still full
        assert buf.is_full
    
    def test_extend(self):
        """Test extend method."""
        buf = RingBuffer(3)
        buf.extend([1, 2, 3, 4, 5])
        
        # Should keep last 3
        assert buf.snapshot() == [3, 4, 5]
    
    def test_thread_safety(self):
        """Test thread safety with concurrent appends."""
        buf = RingBuffer(100)
        
        def writer():
            for i in range(50):
                buf.append(i)
        
        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(buf) == 100
        snapshot = buf.snapshot()
        assert len(snapshot) == 100
        assert None not in snapshot