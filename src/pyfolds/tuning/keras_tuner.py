"""Keras Tuner integration for TensorFlow-compatible PyFolds components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _require_keras_tuner():
    try:
        import keras_tuner as kt
    except ImportError as exc:
        raise ImportError("Dependência opcional ausente: `keras-tuner`.") from exc
    return kt


def run_hyperparameter_search(
    build_model_fn: Callable[[Any], Any],
    train_data: Any,
    val_data: Any,
    *,
    objective: str = "val_loss",
    max_trials: int = 5,
    executions_per_trial: int = 1,
    epochs: int = 3,
    project_dir: str = "./outputs/keras_tuner",
    project_name: str = "pyfolds_hybrid",
) -> Dict[str, Any]:
    kt = _require_keras_tuner()

    Path(project_dir).mkdir(parents=True, exist_ok=True)
    tuner = kt.RandomSearch(
        build_model_fn,
        objective=objective,
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=project_dir,
        project_name=project_name,
    )
    tuner.search(train_data, validation_data=val_data, epochs=epochs)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)
    return {"tuner": tuner, "best_hyperparameters": best_hp[0] if best_hp else None}
