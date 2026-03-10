"""Compatibility wrapper exporting the Noetic LEIBREG bridge."""

import importlib


class NoeticLeibregBridge:
    """Lazy constructor proxy for ``noetic.integration.PyFoldsBridge``."""

    def __new__(cls, *args, **kwargs):
        bridge_cls = importlib.import_module("noetic.integration").PyFoldsBridge
        return bridge_cls(*args, **kwargs)


__all__ = ["NoeticLeibregBridge"]
