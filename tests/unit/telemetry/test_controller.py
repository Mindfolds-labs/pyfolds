"""Tests for TelemetryController."""

import pytest
import pyfolds


class TestTelemetryController:
    """Test telemetry controller."""
    
    def test_initialization(self):
        """Test initialization."""
        cfg = pyfolds.TelemetryConfig(profile="light")
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.enabled() is True
    
    def test_should_emit(self):
        """Test emission logic."""
        cfg = pyfolds.TelemetryConfig(profile="light", sample_every=10)
        telem = pyfolds.TelemetryController(cfg)
        
        assert telem.should_emit(0) is True
        assert telem.should_emit(5) is False
        assert telem.should_emit(10) is True
