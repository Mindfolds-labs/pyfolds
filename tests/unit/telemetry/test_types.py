"""Tests for telemetry types."""

from pyfolds.telemetry import ForwardPayload, CommitPayload, SleepPayload


class TestPayloadTypes:
    """Test payload types."""
    
    def test_forward_payload(self):
        """Test forward payload type."""
        payload: ForwardPayload = {
            'spike_rate': 0.15,
            'theta': 4.5,
            'r_hat': 0.1,
            'v_dend_mean': 3.2,
            'u_mean': 2.1,
            'saturation_ratio': 0.05,
            'N_mean': 12.3,
            'I_mean': 2.5,
            'W_mean': 0.8,
            'duration_ms': 1.2
        }
        
        assert payload['spike_rate'] == 0.15
        assert payload['theta'] == 4.5
        assert payload['duration_ms'] == 1.2
    
    def test_commit_payload(self):
        """Test commit payload type."""
        payload: CommitPayload = {
            'post_rate': 0.15,
            'R': 0.8,
            'delta_N_mean': 0.3,
            'delta_I_mean': 0.1,
            'synapses_updated': 128,
            'duration_ms': 2.5
        }
        
        assert payload['post_rate'] == 0.15
        assert payload['R'] == 0.8
        assert payload['synapses_updated'] == 128
    
    def test_sleep_payload(self):
        """Test sleep payload type."""
        payload: SleepPayload = {
            'duration': 60.0,
            'N_mean_before': 15.2,
            'N_mean_after': 15.5,
            'I_mean_before': 3.2,
            'I_mean_after': 2.8
        }
        
        assert payload['duration'] == 60.0
        assert payload['N_mean_before'] == 15.2
        assert payload['N_mean_after'] == 15.5