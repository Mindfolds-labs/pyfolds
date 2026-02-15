import pytest

from pyfolds.wave import MPJRDWaveConfig


def test_wave_config_defaults_valid():
    cfg = MPJRDWaveConfig()
    assert cfg.wave_enabled is True
    assert cfg.base_frequency > 0


def test_wave_config_rejects_invalid_buffer():
    with pytest.raises(ValueError):
        MPJRDWaveConfig(phase_buffer_size=0)
