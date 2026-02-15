"""Tests for telemetry types."""

from pyfolds.telemetry import ForwardPayload, CommitPayload, SleepPayload


class TestPayloadTypes:
    """Test payload types."""
    
    def test_forward_payload(self):
        """Test forward payload."""
        payload: ForwardPayload = {
            'spike_rate': 0.15,
            'theta': 4.5
        }
        
        assert payload['spike_rate'] == 0.15
        assert payload['theta'] == 4.5
