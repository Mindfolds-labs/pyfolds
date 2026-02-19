"""TensorFlow layer wrappers for MPJRD recurrent cells."""

from __future__ import annotations

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - import-guarded by package __init__
    raise ImportError(
        "TensorFlow backend requested, but TensorFlow is not installed. "
        "Install it with `pip install tensorflow` (or `tensorflow-cpu`) to use `pyfolds.tf`."
    ) from exc

from .neuron import MPJRDTFNeuronCell


class MPJRDTFLayer(tf.keras.layers.Layer):
    """Keras-compatible wrapper that exposes torch-like dict output."""

    def __init__(
        self,
        units: int,
        *,
        return_sequences: bool = True,
        return_state: bool = False,
        **cell_kwargs,
    ) -> None:
        super().__init__()
        self.cell = MPJRDTFNeuronCell(units=units, **cell_kwargs)
        self.rnn = tf.keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
        )

    def call(self, inputs, initial_state=None, training=None):
        out = self.rnn(inputs, initial_state=initial_state, training=training)
        if self.rnn.return_state:
            spikes, *states = out
            return {"spikes": spikes, "state": states[0]}
        return {"spikes": out}
