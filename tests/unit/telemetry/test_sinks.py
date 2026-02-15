"""Tests for telemetry sinks."""

import pytest
import json
import pyfolds


class TestMemorySink:
    """Test memory sink."""
    
    def test_emit(self):
        """Test emit stores events."""
        sink = pyfolds.MemorySink(5)
        
        for i in range(3):
            sink.emit(pyfolds.forward_event(i, "online"))
        
        assert len(sink.snapshot()) == 3
    
    def test_capacity_limit(self):
        """Test that sink respects capacity limit."""
        sink = pyfolds.MemorySink(2)
        
        sink.emit(pyfolds.forward_event(1, "online"))
        sink.emit(pyfolds.forward_event(2, "online"))
        sink.emit(pyfolds.forward_event(3, "online"))  # Should overwrite oldest
        
        snapshot = sink.snapshot()
        assert len(snapshot) == 2
        # Should keep most recent 2
        assert snapshot[0].step_id == 2
        assert snapshot[1].step_id == 3
    
    def test_clear(self):
        """Test clear method."""
        sink = pyfolds.MemorySink(5)
        
        sink.emit(pyfolds.forward_event(1, "online"))
        sink.emit(pyfolds.forward_event(2, "online"))
        
        assert len(sink.snapshot()) == 2
        
        sink.clear()
        assert len(sink.snapshot()) == 0


class TestConsoleSink:
    """Test console sink."""
    
    def test_verbose(self, capsys):
        """Test verbose mode prints payload."""
        sink = pyfolds.ConsoleSink(verbose=True)
        
        event = pyfolds.forward_event(
            step_id=42,
            mode="online",
            spike_rate=0.15,
            theta=4.5
        )
        sink.emit(event)
        
        captured = capsys.readouterr()
        assert "step=42" in captured.out
        assert "spike_rate" in captured.out
        assert "theta" in captured.out
    
    def test_non_verbose(self, capsys):
        """Test non-verbose mode only prints step and phase."""
        sink = pyfolds.ConsoleSink(verbose=False)
        
        event = pyfolds.forward_event(
            step_id=42,
            mode="online",
            spike_rate=0.15,
            theta=4.5
        )
        sink.emit(event)
        
        captured = capsys.readouterr()
        assert "step=42" in captured.out
        assert "forward" in captured.out
        assert "spike_rate" not in captured.out  # Should not show payload


class TestJSONLinesSink:
    """Test JSON lines sink."""
    
    def test_emit_basic(self, tmp_path):
        """Test basic emit to file."""
        log_file = tmp_path / "test.jsonl"
        
        with pyfolds.JSONLinesSink(log_file) as sink:
            sink.emit(pyfolds.forward_event(1, "online", spike_rate=0.15))
        
        assert log_file.exists()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) == 1
        
        data = json.loads(lines[0])
        assert data['step_id'] == 1
        assert data['mode'] == "online"
        assert data['payload']['spike_rate'] == 0.15
    
    def test_truncate_mode(self, tmp_path):
        """Test truncate=True overwrites file."""
        path = tmp_path / "test.jsonl"
        
        # First write
        with pyfolds.JSONLinesSink(path, truncate=True) as sink:
            sink.emit(pyfolds.forward_event(1, "online"))
        
        # Second write with truncate (should overwrite)
        with pyfolds.JSONLinesSink(path, truncate=True) as sink:
            sink.emit(pyfolds.forward_event(2, "online"))
        
        lines = path.read_text().strip().split('\n')
        assert len(lines) == 1
        assert json.loads(lines[0])['step_id'] == 2
    
    def test_append_mode(self, tmp_path):
        """Test truncate=False appends to file."""
        path = tmp_path / "test.jsonl"
        
        # First write
        with pyfolds.JSONLinesSink(path, truncate=True) as sink:
            sink.emit(pyfolds.forward_event(1, "online"))
        
        # Second write with append
        with pyfolds.JSONLinesSink(path, truncate=False) as sink:
            sink.emit(pyfolds.forward_event(2, "online"))
        
        lines = path.read_text().strip().split('\n')
        assert len(lines) == 2
        assert json.loads(lines[0])['step_id'] == 1
        assert json.loads(lines[1])['step_id'] == 2
    
    def test_tensor_serialization(self, tmp_path):
        """Test serialization of PyTorch tensors."""
        pytest.importorskip("torch")
        import torch
        
        path = tmp_path / "test.jsonl"
        
        with pyfolds.JSONLinesSink(path, truncate=True) as sink:
            event = pyfolds.forward_event(
                step_id=0,
                mode="online",
                scalar_tensor=torch.tensor(5.0),
                array_tensor=torch.tensor([1, 2, 3]),
                matrix_tensor=torch.ones(2, 2)
            )
            sink.emit(event)
        
        lines = path.read_text().strip().split('\n')
        data = json.loads(lines[0])
        
        # Scalar tensor becomes number
        assert data['payload']['scalar_tensor'] == 5.0
        
        # 1D tensor becomes list
        assert data['payload']['array_tensor'] == [1, 2, 3]
        
        # 2D tensor becomes list of lists
        assert data['payload']['matrix_tensor'] == [[1.0, 1.0], [1.0, 1.0]]
    
    def test_non_serializable_fallback(self, tmp_path):
        """Test fallback for non-serializable objects."""
        path = tmp_path / "test.jsonl"
        
        class Unserializable:
            def __str__(self):
                return "unserializable_object"
        
        with pyfolds.JSONLinesSink(path, truncate=True) as sink:
            event = pyfolds.forward_event(
                step_id=0,
                mode="online",
                bad_obj=Unserializable()
            )
            sink.emit(event)
        
        lines = path.read_text().strip().split('\n')
        data = json.loads(lines[0])
        assert data['payload']['bad_obj'] == "unserializable_object"


class TestDistributorSink:
    """Test distributor sink."""
    
    def test_distributes_to_all_sinks(self, tmp_path):
        """Test that events go to all sinks."""
        mem_sink = pyfolds.MemorySink(5)
        path = tmp_path / "dist.jsonl"
        json_sink = pyfolds.JSONLinesSink(path, truncate=True)
        
        dist = pyfolds.DistributorSink([mem_sink, json_sink])
        
        event = pyfolds.forward_event(42, "online", spike_rate=0.15)
        dist.emit(event)
        dist.flush()
        
        # Check memory sink
        assert len(mem_sink.snapshot()) == 1
        
        # Check JSON sink
        lines = path.read_text().strip().split('\n')
        assert len(lines) == 1
        assert json.loads(lines[0])['step_id'] == 42
    
    def test_sink_failure_doesnt_affect_others(self):
        """Test that one sink failing doesn't stop others."""
        
        class FailingSink(pyfolds.Sink):
            def emit(self, event):
                raise RuntimeError("Intentional failure")
        
        mem_sink = pyfolds.MemorySink(10)
        failing = FailingSink()
        
        dist = pyfolds.DistributorSink([failing, mem_sink])
        
        event = pyfolds.forward_event(0, "online")
        
        # Should not raise despite failing sink
        dist.emit(event)
        
        # Memory sink should still receive event
        assert len(mem_sink.snapshot()) == 1