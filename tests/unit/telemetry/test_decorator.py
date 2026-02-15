"""Tests for telemetry decorator."""

import pytest
import pyfolds


class TestTelemetryDecorator:
    """Test decorator."""
    
    def test_decorator(self):
        """Test decorator functionality."""
        
        class TestNeuron:
            def __init__(self):
                self.telemetry = pyfolds.TelemetryController(
                    pyfolds.TelemetryConfig(profile="heavy")
                )
                self.step_id = 0
                self.mode = "online"
            
            @pyfolds.telemetry(phase="forward", capture_return=True)
            def forward(self, x):
                return {'spike_rate': 0.15}
        
        neuron = TestNeuron()
        result = neuron.forward(10)
        
        assert result['spike_rate'] == 0.15
        assert len(neuron.telemetry.snapshot()) == 1
