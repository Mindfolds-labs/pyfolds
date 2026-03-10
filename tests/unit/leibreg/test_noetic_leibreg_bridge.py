import pytest


def test_bridge_import_surface() -> None:
    pytest.importorskip("noetic.integration")
    from pyfolds.leibreg.leibreg_bridge import NoeticLeibregBridge

    assert callable(NoeticLeibregBridge)
