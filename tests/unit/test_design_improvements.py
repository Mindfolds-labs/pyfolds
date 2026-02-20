import pytest

from pyfolds.advanced import MPJRDNeuronAdvanced
from pyfolds import NeuronConfig
from pyfolds.core.base import BaseNeuron
from pyfolds.factory import NeuronFactory, NeuronType
from pyfolds.network import NetworkBuilder
from pyfolds.utils import LearningMode
from pyfolds.utils.context import learning_mode


class DummyNeuron(BaseNeuron):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.mode = LearningMode.ONLINE

    def forward(self, x, *args, **kwargs):
        return {"spikes": x}

    def set_mode(self, mode: LearningMode) -> None:
        self.mode = mode

    def apply_plasticity(self, dt: float = 1.0, **kwargs) -> None:
        return None

    def get_metrics(self):
        return {"dummy": True}


def test_factory_creates_builtin_types():
    cfg = NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)

    basic = NeuronFactory.create(NeuronType.BASIC, cfg)
    advanced = NeuronFactory.create(NeuronType.ADVANCED, cfg)

    assert isinstance(basic, BaseNeuron)
    assert isinstance(advanced, MPJRDNeuronAdvanced)


def test_factory_custom_registry_and_unknown_type():
    cfg = NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    NeuronFactory.register("dummy", DummyNeuron)

    custom = NeuronFactory.create("dummy", cfg)
    assert isinstance(custom, DummyNeuron)

    with pytest.raises(ValueError, match="desconhecido"):
        NeuronFactory.create("nao-existe", cfg)


def test_learning_mode_context_restores_even_on_error():
    cfg = NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)
    neuron = NeuronFactory.create(NeuronType.BASIC, cfg)

    assert neuron.mode == LearningMode.ONLINE

    with pytest.raises(RuntimeError):
        with learning_mode(neuron, LearningMode.BATCH):
            assert neuron.mode == LearningMode.BATCH
            raise RuntimeError("forced")

    assert neuron.mode == LearningMode.ONLINE


def test_network_builder_connects_layers_and_builds():
    cfg = NeuronConfig(n_dendrites=2, n_synapses_per_dendrite=4)

    net = (
        NetworkBuilder("builder_net")
        .add_layer("input", n_neurons=3, cfg=cfg)
        .add_layer("hidden", n_neurons=2, cfg=cfg)
        .add_layer("output", n_neurons=1, cfg=cfg)
        .build()
    )

    assert net.built is True
    assert net.connections == [("input", "hidden"), ("hidden", "output")]
    assert net.input_layer == "input"
    assert net.output_layer == "output"
