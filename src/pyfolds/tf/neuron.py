"""TensorFlow neuron primitives compatible with ``keras.layers.RNN``."""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - import-guarded by package __init__
    raise ImportError(
        "TensorFlow backend requested, but TensorFlow is not installed. "
        "Install it with `pip install tensorflow` (or `tensorflow-cpu`) to use `pyfolds.tf`."
    ) from exc


class MPJRDTFNeuronCell(tf.keras.layers.Layer):
    """Minimal recurrent cell implementing a membrane/spike update step."""

    def __init__(
        self,
        units: int,
        *,
        threshold: float = 1.0,
        decay: float = 0.9,
        dt: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError("units deve ser > 0")
        self.units = int(units)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.default_dt = float(dt)

    @property
    def state_size(self) -> int:
        return self.units

    @property
    def output_size(self) -> int:
        return self.units

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32

        if batch_size is None:
            if inputs is None:
                raise ValueError("batch_size ou inputs deve ser informado")
            batch_size = tf.shape(inputs)[0]

        return [tf.zeros((batch_size, self.units), dtype=dtype)]

    def step(
        self,
        inputs: tf.Tensor,
        prev_state: tf.Tensor,
        *,
        dt: Optional[float] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        dt_value = self.default_dt if dt is None else float(dt)
        integrated = (self.decay * prev_state) + (inputs * dt_value)
        spikes = tf.cast(integrated >= self.threshold, integrated.dtype)
        new_state = integrated - (spikes * self.threshold)
        return spikes, new_state

    def call(self, inputs, states, training=None, dt=None):
        del training
        prev_state = states[0]
        spikes, new_state = self.step(inputs, prev_state, dt=dt)
        return spikes, [new_state]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "threshold": self.threshold,
                "decay": self.decay,
                "dt": self.default_dt,
            }
        )
        return config
