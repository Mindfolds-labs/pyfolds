"""Tests for ShortTermDynamicsMixin."""

import pytest
import torch
import pyfolds


class TestShortTermDynamicsMixin:
    """Test short-term plasticity (STP)."""
    
    def test_initialization(self, full_config):
        """Test STP parameters."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        D = full_config.n_dendrites
        S = full_config.n_synapses_per_dendrite
        
        assert neuron.u_stp.shape == (D, S)
        assert neuron.R_stp.shape == (D, S)
        assert torch.allclose(neuron.u_stp, torch.ones(D, S) * full_config.u0)
    
    def test_update_no_spikes(self, full_config):
        """Test decay without spikes."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        D = full_config.n_dendrites
        S = full_config.n_synapses_per_dendrite
        
        neuron.u_stp.fill_(0.5)
        neuron.R_stp.fill_(0.8)
        
        x = torch.zeros(2, D, S)
        
        import math
        decay_fac = math.exp(-1.0 / neuron.tau_fac)
        
        neuron._update_short_term_dynamics(x, dt=1.0)
        
        expected_u = torch.ones(D, S) * 0.5 * decay_fac
        assert torch.allclose(neuron.u_stp, expected_u, rtol=1e-4)

        decay_rec = math.exp(-1.0 / neuron.tau_rec)
        expected_R = torch.ones(D, S) * (0.8 * decay_rec + (1 - 0.8) * (1 - decay_rec))
        assert torch.allclose(neuron.R_stp, expected_R, rtol=1e-4)
    
    def test_update_with_spikes(self, full_config):
        """Test facilitation with spikes."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
            
        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        D = full_config.n_dendrites
        S = full_config.n_synapses_per_dendrite
        
        neuron.u_stp.fill_(0.2)
        
        x = torch.zeros(2, D, S)
        x[:, 0, 0] = 1.0  # Spike at (0,0)
        
        import math
        decay_fac = math.exp(-1.0 / neuron.tau_fac)
        
        neuron._update_short_term_dynamics(x, dt=1.0)
        
        # u = u*decay + U*(1-u)
        expected_u_spike = 0.2 * decay_fac + full_config.U * (1 - 0.2)
        expected_u_no_spike = 0.2 * decay_fac
        
        assert abs(neuron.u_stp[0, 0].item() - expected_u_spike) < 1e-4
        assert abs(neuron.u_stp[0, 1].item() - expected_u_no_spike) < 1e-4
    def test_tsodyks_markram_recovery(self, full_config):
        """R deve recuperar para próximo de 1 sem estímulo."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        neuron.R_stp.fill_(0.2)

        x = torch.zeros(1, full_config.n_dendrites, full_config.n_synapses_per_dendrite)
        for _ in range(1000):
            neuron._update_short_term_dynamics(x, dt=1.0)

        assert 0.45 < neuron.R_stp.mean().item() < 0.55


    def test_update_aligns_stp_buffers_to_input_device(self, full_config):
        """Buffers de STP devem acompanhar device da entrada para evitar mismatch."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config).cpu()
        x = torch.zeros(2, full_config.n_dendrites, full_config.n_synapses_per_dendrite, device='cuda')

        neuron._update_short_term_dynamics(x, dt=1.0)

        assert neuron.u_stp.device.type == 'cuda'
        assert neuron.R_stp.device.type == 'cuda'


    def test_stp_buffers_remain_registered_after_device_alignment(self, full_config):
        """Realinhamento de device não deve remover buffers do state_dict."""
        if not pyfolds.ADVANCED_AVAILABLE:
            pytest.skip("Advanced module not available")

        neuron = pyfolds.MPJRDNeuronAdvanced(full_config)
        neuron._align_short_term_buffers_device(torch.device('cpu'))

        state = neuron.state_dict()
        assert 'u_stp' in state
        assert 'R_stp' in state
