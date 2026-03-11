"""TensorFlow neuron primitives compatible with ``keras.layers.RNN``.

Notes
-----
The TensorFlow cell intentionally supports a reduced set of integration modes
compared to the PyTorch backend. At the moment, only ``wta_soft_approx`` is
implemented for dendritic aggregation.
"""

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
    """Minimal recurrent cell implementing a membrane/spike update step.

    The cell accepts either flat inputs ``[B, U]`` (legacy behavior) or
    structured dendritic inputs:

    - ``[B, D, S]``: one neuron view with ``D`` dendrites and ``S`` synapses.
    - ``[B, U, D, S]``: per-unit structured dendrites.

    ``integration_mode`` in TensorFlow is intentionally constrained and does not
    replicate all PyTorch modes 1:1.

    State layout
    ------------
    The recurrent state is composite and represented as three tensors to remain
    compatible with ``keras.layers.RNN`` nested-state support:

    1. ``membrane``: soma membrane potential ``[B, U]``.
    2. ``adapt_state``: lightweight adaptation trace ``[B, U]`` persisted across steps.
    3. ``context_state``: minimal scalar context ``[B, 1]`` persisted across steps.

    Reset behavior
    --------------
    ``get_initial_state`` resets all three state tensors to zeros. During each
    ``call``/``step``, only these tensors are persisted between time steps; masks,
    thresholds and static configuration are immutable layer attributes.
    """

    def __init__(
        self,
        units: int,
        *,
        threshold: float = 1.0,
        decay: float = 0.9,
        dt: float = 1.0,
        integration_mode: str = "wta_soft_approx",
        dendritic_threshold: float = 0.0,
        dendritic_gate: str = "sigmoid",
        clip_value: float = 10.0,
        connectivity_mask: Optional[tf.Tensor] = None,
        pruning_mask: Optional[tf.Tensor] = None,
        edge_inference_mode: bool = False,
        plasticity_mode: str = "none",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if units <= 0:
            raise ValueError("Invalid argument `units`: expected a positive integer.")
        self.units = int(units)
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.default_dt = float(dt)
        requested_integration_mode = str(integration_mode)
        self.integration_mode = (
            requested_integration_mode
            if requested_integration_mode == "wta_soft_approx"
            else "wta_soft_approx"
        )
        self.requested_integration_mode = requested_integration_mode
        self.dendritic_threshold = float(dendritic_threshold)
        self.dendritic_gate = str(dendritic_gate)
        self.clip_value = float(clip_value)
        self.connectivity_mask = connectivity_mask
        self.pruning_mask = pruning_mask
        self.edge_inference_mode = bool(edge_inference_mode)
        # TensorFlow backend has no online plasticity; force-disable silently on edge.
        self.plasticity_mode = "none" if self.edge_inference_mode else str(plasticity_mode)
        self._last_diagnostics = {}
        self._integration_mode_blocked = requested_integration_mode != self.integration_mode
        if self.dendritic_gate not in {"sigmoid", "hard"}:
            raise ValueError("Invalid argument `dendritic_gate`: use `sigmoid` or `hard`.")
        if self.clip_value <= 0:
            raise ValueError("Invalid argument `clip_value`: expected a value > 0.")

    @property
    def state_size(self):
        return (self.units, self.units, 1)

    @property
    def output_size(self) -> int:
        return self.units

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if dtype is None:
            dtype = tf.float32

        if batch_size is None:
            if inputs is None:
                raise ValueError("Provide `batch_size` or `inputs` to infer initial state shape.")
            batch_size = tf.shape(inputs)[0]

        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, 1), dtype=dtype),
        ]

    def step(
        self,
        inputs: tf.Tensor,
        prev_membrane: tf.Tensor,
        prev_adapt_state: tf.Tensor,
        prev_context_state: tf.Tensor,
        *,
        dt: Optional[float] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        dt_value = self.default_dt if dt is None else float(dt)
        if dt_value <= 0:
            raise ValueError("Invalid argument `dt`: expected a value > 0.")

        soma_current, v_dend = self._compute_somatic_current(inputs)
        integrated = (self.decay * prev_membrane) + (soma_current * dt_value)
        integrated = tf.clip_by_value(integrated, -self.clip_value, self.clip_value)

        next_adapt_state = tf.clip_by_value(
            (0.95 * prev_adapt_state) + (0.05 * tf.abs(integrated)),
            0.0,
            self.clip_value,
        )
        context_signal = tf.reduce_mean(integrated, axis=-1, keepdims=True)
        next_context_state = tf.clip_by_value(
            (0.9 * prev_context_state) + (0.1 * context_signal),
            -self.clip_value,
            self.clip_value,
        )

        theta_eff = tf.fill(tf.shape(integrated), tf.cast(self.threshold, integrated.dtype))
        spikes = tf.cast(integrated >= theta_eff, integrated.dtype)
        new_state = integrated - (spikes * theta_eff)
        new_state = tf.clip_by_value(new_state, -self.clip_value, self.clip_value)

        diagnostics = {"u": integrated, "theta_eff": theta_eff}
        if v_dend is not None:
            diagnostics["v_dend"] = v_dend
        diagnostics["adapt_state"] = next_adapt_state
        diagnostics["context_state"] = next_context_state
        diagnostics["edge_inference_mode"] = self.edge_inference_mode
        diagnostics["plasticity_mode"] = self.plasticity_mode
        diagnostics["integration_mode_blocked"] = self._integration_mode_blocked
        self._last_diagnostics = diagnostics

        if return_diagnostics:
            return spikes, new_state, next_adapt_state, next_context_state, diagnostics
        return spikes, new_state, next_adapt_state, next_context_state

    @property
    def last_diagnostics(self):
        """Diagnostics from the most recent ``step`` execution."""
        return self._last_diagnostics

    def _apply_masks(self, x: tf.Tensor) -> tf.Tensor:
        if self.connectivity_mask is not None:
            mask = tf.cast(self.connectivity_mask, x.dtype)
            x = tf.where(mask > 0, x, tf.zeros_like(x))
        if self.pruning_mask is not None:
            mask = tf.cast(self.pruning_mask, x.dtype)
            x = x * mask
        return x

    def _gate_dendrites(self, v_dend: tf.Tensor) -> tf.Tensor:
        if self.dendritic_gate == "hard":
            gate = tf.where(
                v_dend >= tf.cast(self.dendritic_threshold, v_dend.dtype),
                tf.ones_like(v_dend),
                tf.zeros_like(v_dend),
            )
        else:
            gate = tf.sigmoid(v_dend - tf.cast(self.dendritic_threshold, v_dend.dtype))
        return gate * v_dend

    def _aggregate_dendrites(self, v_dend_gated: tf.Tensor, axis: int) -> tf.Tensor:
        # ``wta_soft_approx``: softmax-weighted approximation of winner-take-all.
        alpha = tf.nn.softmax(v_dend_gated, axis=axis)
        return tf.reduce_sum(alpha * v_dend_gated, axis=axis)

    def _compute_somatic_current(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        x = tf.convert_to_tensor(inputs)
        rank = x.shape.rank

        if rank == 2:
            x_masked = self._apply_masks(x)
            x_clip = tf.clip_by_value(x_masked, -self.clip_value, self.clip_value)
            return x_clip, None

        if rank == 3:
            x_masked = self._apply_masks(x)
            v_dend = tf.reduce_sum(x_masked, axis=-1)  # [B, D]
            v_dend = tf.clip_by_value(v_dend, -self.clip_value, self.clip_value)
            v_dend_gated = self._gate_dendrites(v_dend)
            soma_current = self._aggregate_dendrites(v_dend_gated, axis=1)  # [B]
            soma_current = tf.expand_dims(soma_current, axis=-1)
            soma_current = tf.broadcast_to(soma_current, (tf.shape(x)[0], self.units))
            return soma_current, v_dend

        if rank == 4:
            x_masked = self._apply_masks(x)
            v_dend = tf.reduce_sum(x_masked, axis=-1)  # [B, U, D]
            v_dend = tf.clip_by_value(v_dend, -self.clip_value, self.clip_value)
            v_dend_gated = self._gate_dendrites(v_dend)
            soma_current = self._aggregate_dendrites(v_dend_gated, axis=2)  # [B, U]
            return soma_current, v_dend

        raise ValueError(
            "Unsupported input rank for `MPJRDTFNeuronCell.step`: expected [B, U], [B, D, S], "
            "or [B, U, D, S]."
        )

    def call(self, inputs, states, training=None, dt=None):
        del training
        prev_membrane, prev_adapt_state, prev_context_state = states
        spikes, new_membrane, new_adapt_state, new_context_state = self.step(
            inputs,
            prev_membrane,
            prev_adapt_state,
            prev_context_state,
            dt=dt,
        )
        return spikes, [new_membrane, new_adapt_state, new_context_state]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "threshold": self.threshold,
                "decay": self.decay,
                "dt": self.default_dt,
                "integration_mode": self.integration_mode,
                "requested_integration_mode": self.requested_integration_mode,
                "dendritic_threshold": self.dendritic_threshold,
                "dendritic_gate": self.dendritic_gate,
                "clip_value": self.clip_value,
                "edge_inference_mode": self.edge_inference_mode,
                "plasticity_mode": self.plasticity_mode,
            }
        )
        return config
