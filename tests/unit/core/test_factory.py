import pyfolds

from pyfolds.core import MPJRDConfig, create_neuron
from pyfolds.core.factory import NeuronFactory, NeuronType
from pyfolds.wave import MPJRDWaveConfig, MPJRDWaveNeuron


def test_create_neuron_infers_standard_type():
    neuron = create_neuron(MPJRDConfig())
    assert isinstance(neuron, pyfolds.MPJRDNeuron)


def test_create_neuron_infers_wave_type():
    neuron = create_neuron(MPJRDWaveConfig())
    assert isinstance(neuron, MPJRDWaveNeuron)


def test_factory_raises_for_unregistered_type():
    cfg = MPJRDConfig()
    NeuronFactory._registry.clear()

    try:
        NeuronFactory.create(NeuronType.STANDARD, cfg)
        assert False, "Era esperado ValueError"
    except ValueError as exc:
        assert "não registrado" in str(exc)
    finally:
        from pyfolds.core.factory import register_default_neurons

        register_default_neurons()
