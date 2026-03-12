import pytest


def test_bridge_import_surface() -> None:
    pytest.importorskip("noetic.integration")
    from pyfolds.leibreg.leibreg_bridge import NoeticLeibregBridge

    assert callable(NoeticLeibregBridge)


def test_bridge_raises_clear_error_when_noetic_missing(monkeypatch) -> None:
    import importlib

    original = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "noetic.integration":
            raise ModuleNotFoundError("No module named 'noetic'")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    from pyfolds.leibreg.leibreg_bridge import NoeticLeibregBridge

    with pytest.raises(ImportError, match="optional dependency `noetic` is not installed"):
        NoeticLeibregBridge()
