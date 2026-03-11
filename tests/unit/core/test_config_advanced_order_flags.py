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
    assert source.count("wave_enabled:") == 1
