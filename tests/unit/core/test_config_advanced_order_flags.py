from dataclasses import fields

import pyfolds
import pytest
from pyfolds import NeuronConfig


def test_accepts_stdp_source_and_ltd_rule_flags():
    cfg = NeuronConfig(stdp_input_source="raw", ltd_rule="classic")
    assert cfg.stdp_input_source == "raw"
    assert cfg.ltd_rule == "classic"


@pytest.mark.parametrize("kwargs", [
    {"stdp_input_source": "invalid"},
    {"ltd_rule": "invalid"},
])
def test_rejects_invalid_order_flags(kwargs):
    with pytest.raises(ValueError):
        NeuronConfig(**kwargs)


def test_config_no_duplicate_fields():
    names = [f.name for f in fields(pyfolds.NeuronConfig)]
    assert len(names) == len(set(names))

    import inspect

    source = inspect.getsource(pyfolds.NeuronConfig)
    assert source.count("\n    wave_enabled:") == 1


def test_experimental_flags_defaults_and_roundtrip_serialization():
    cfg = NeuronConfig()

    assert cfg.experimental_wave_enabled is True
    assert cfg.experimental_circadian_enabled is True
    assert cfg.experimental_engram_enabled is True
    assert cfg.experimental_engram_indexing_enabled is True
    assert cfg.experimental_engram_cache_enabled is True

    payload = cfg.to_dict()
    rebuilt = NeuronConfig.from_dict(payload)

    assert rebuilt.experimental_wave_enabled is True
    assert rebuilt.experimental_circadian_enabled is True
    assert rebuilt.experimental_engram_enabled is True
    assert rebuilt.experimental_engram_indexing_enabled is True
    assert rebuilt.experimental_engram_cache_enabled is True


@pytest.mark.parametrize("field_name", [
    "experimental_wave_enabled",
    "experimental_circadian_enabled",
    "experimental_engram_enabled",
    "experimental_engram_indexing_enabled",
    "experimental_engram_cache_enabled",
])
def test_experimental_flags_require_bool(field_name):
    with pytest.raises(ValueError, match=f"{field_name} must be bool"):
        NeuronConfig(**{field_name: 1})
