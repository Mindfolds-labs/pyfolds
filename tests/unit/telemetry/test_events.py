"""Tests for telemetry events."""

import pyfolds


class TestTelemetryEvents:
    """Test events."""
    
    def test_forward_event(self):
        """Test forward event."""
        event = pyfolds.forward_event(
            step_id=42,
            mode="online",
            spike_rate=0.15
        )
        
        assert event.step_id == 42
        assert event.phase == "forward"
        assert event.payload['spike_rate'] == 0.15
