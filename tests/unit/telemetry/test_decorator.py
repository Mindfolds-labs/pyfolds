"""Tests for telemetry decorator."""

import pytest
import pyfolds


class TestTelemetryDecorator:
    """Test decorator."""
    
    def test_decorator_basic(self):
        """Test basic decorator functionality."""
        
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
    
    def test_decorator_with_sample_rate(self):
        """Test decorator with sampling."""
        
        class TestNeuron:
            def __init__(self):
                self.telemetry = pyfolds.TelemetryController(
                    pyfolds.TelemetryConfig(profile="heavy")
                )
                self.step_id = 0
                self.mode = "online"
                self.call_count = 0
            
            @pyfolds.telemetry(phase="forward", sample_rate=0.5)
            def forward(self, x):
                self.call_count += 1
                return {'spike_rate': 0.15}
        
        neuron = TestNeuron()
        
        # Run many times
        for _ in range(1000):
            neuron.forward(10)
        
        # A função decorada sempre deve executar; amostragem só afeta emissão
        assert neuron.call_count == 1000

        emitted = len(neuron.telemetry.snapshot())
        assert 400 < emitted < 600
    
    def test_decorator_lazy_evaluation(self):
        """Test that decorator works with lazy events."""
        
        class TestNeuron:
            def __init__(self):
                self.telemetry = pyfolds.TelemetryController(
                    pyfolds.TelemetryConfig(profile="heavy")
                )
                self.step_id = 0
                self.mode = "online"
                self.expensive_calls = 0
            
            def get_spike_rate(self):
                self.expensive_calls += 1
                return 0.15
            
            @pyfolds.telemetry(phase="forward")
            def forward(self, x):
                # This should use lazy evaluation internally
                return {'spike_rate': self.get_spike_rate()}
        
        neuron = TestNeuron()
        
        # First call - should evaluate
        result = neuron.forward(10)
        assert result['spike_rate'] == 0.15
        assert neuron.expensive_calls == 1
        
        # Telemetry should have captured it
        assert len(neuron.telemetry.snapshot()) == 1