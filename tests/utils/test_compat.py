import pytest

from pyfolds.utils.compat import has_tensorboard, require_tensorboard, OptionalDependencyError


def test_has_tensorboard_returns_bool():
    assert isinstance(has_tensorboard(), bool)


def test_require_tensorboard_actionable_message(monkeypatch):
    monkeypatch.setattr("pyfolds.utils.compat._import_optional", lambda name: None)
    with pytest.raises(OptionalDependencyError) as exc:
        require_tensorboard()
    assert "pyfolds[telemetry]" in str(exc.value)
