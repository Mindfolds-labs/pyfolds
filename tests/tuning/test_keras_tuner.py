import pytest

from pyfolds.tuning.keras_tuner import run_hyperparameter_search


def test_keras_tuner_missing_dep():
    with pytest.raises(ImportError):
        run_hyperparameter_search(lambda hp: None, None, None, max_trials=1, epochs=1)
