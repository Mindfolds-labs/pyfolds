"""Tests for telemetry sinks."""

import pytest
import tempfile
from pathlib import Path
import pyfolds


class TestMemorySink:
    """Test memory sink."""
    
    def test_emit(self):
        """Test emit."""
        sink = pyfolds.MemorySink(5)
        
        for i in range(3):
            sink.emit(pyfolds.forward_event(i, "online"))
        
        assert len(sink.snapshot()) == 3


class TestJSONLinesSink:
    """Test JSON lines sink."""
    
    def test_emit(self, tmp_path):
        """Test emit to file."""
        log_file = tmp_path / "test.jsonl"
        
        with pyfolds.JSONLinesSink(log_file) as sink:
            sink.emit(pyfolds.forward_event(1, "online"))
        
        assert log_file.exists()
        assert log_file.read_text().strip() != ""
