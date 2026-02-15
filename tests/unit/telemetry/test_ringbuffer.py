"""Tests for RingBuffer."""

import pytest
import threading
import time
from pyfolds.telemetry import RingBuffer


class TestRingBuffer:
    """Test ring buffer."""
    
    def test_append(self):
        """Test append."""
        buf = RingBuffer(3)
        
        buf.append(1)
        buf.append(2)
        buf.append(3)
        buf.append(4)  # Overwrites oldest
        
        assert buf.snapshot() == [2, 3, 4]
    
    def test_thread_safety(self):
        """Test thread safety."""
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
