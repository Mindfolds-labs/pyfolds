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
