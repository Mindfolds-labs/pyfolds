"""Tests for TelemetryController."""

import pytest
import pyfolds


class TestTelemetryController:
    """Test telemetry controller."""
    
    def test_initialization(self):
        """Test basic initialization."""
        cfg = pyfolds.TelemetryConfig(profile="light")
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.enabled() is True
    
    def test_profile_off(self):
        """Test profile=off disables everything."""
        cfg = pyfolds.TelemetryConfig(profile="off")
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.enabled() is False
        assert telem.should_emit(0) is False
        assert telem.should_emit(100) is False
        assert isinstance(telem.sink, pyfolds.NoOpSink)
    
    def test_profile_light(self):
        """Test profile=light with sampling."""
        cfg = pyfolds.TelemetryConfig(profile="light", sample_every=10)
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.enabled() is True
        assert telem.should_emit(0) is True   # %10 == 0
        assert telem.should_emit(5) is False  # %10 != 0
        assert telem.should_emit(10) is True  # %10 == 0
        assert telem.should_emit(20) is True  # %10 == 0
    
    def test_profile_heavy(self):
        """Test profile=heavy always emits."""
        cfg = pyfolds.TelemetryConfig(profile="heavy")
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.enabled() is True
        assert telem.should_emit(0) is True
        assert telem.should_emit(1) is True
        assert telem.should_emit(100) is True
    
    def test_should_emit_sample(self):
        """Test probabilistic sampling."""
        cfg = pyfolds.TelemetryConfig(profile="light")
        telem = pyfolds.TelemetryController(cfg)
        
        # 100% should always emit
        assert telem.should_emit_sample(1.0) is True
        
        # 0% should never emit
        assert telem.should_emit_sample(0.0) is False
        
        # 50% should emit ~50% of time
        results = [telem.should_emit_sample(0.5) for _ in range(1000)]
        hits = sum(results)
        assert 400 < hits < 600  # Within reasonable range
    
    def test_step_count_increments(self):
        """Test that step counter increments on emit."""
        cfg = pyfolds.TelemetryConfig(profile="heavy")
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.step_count == 0
        
        telem.emit(pyfolds.forward_event(0, "online"))
        assert telem.step_count == 1
        
        telem.emit(pyfolds.forward_event(1, "online"))
        telem.emit(pyfolds.forward_event(2, "online"))
        assert telem.step_count == 3
    
    def test_snapshot_and_clear(self):
        """Test snapshot and clear methods."""
        cfg = pyfolds.TelemetryConfig(profile="heavy")
        telem = pyfolds.TelemetryController(cfg)
        
        # Emit some events
        for i in range(5):
            telem.emit(pyfolds.forward_event(i, "online"))
        
        # Snapshot should have them
        snapshot = telem.snapshot()
        assert len(snapshot) == 5
        
        # Clear should remove them
        telem.clear()
        assert len(telem.snapshot()) == 0
    
    def test_get_stats(self):
        """Test get_stats returns correct structure."""
        cfg = pyfolds.TelemetryConfig(profile="light", sample_every=10)
        telem = pyfolds.TelemetryController(cfg)
        
        stats = telem.get_stats()
        
        assert stats['profile'] == "light"
        assert stats['sample_every'] == 10
        assert stats['step_count'] == 0
        assert stats['sink_type'] == "MemorySink"
        assert 'buffer_size' in stats
        assert 'enabled' in stats
        assert 'estimated_memory_mb' in stats
        assert 'sampling_rate' in stats