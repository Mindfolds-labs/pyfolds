"""Compatibility wrapper exporting the Noetic LEIBREG bridge."""

import importlib


class NoeticLeibregBridge:
    """Lazy constructor proxy for ``noetic.integration.PyFoldsBridge``."""

    def __new__(cls, *args, **kwargs):
        try:
            module = importlib.import_module("noetic.integration")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Noetic bridge requested but optional dependency `noetic` is not installed. "
                "Install it (e.g. `pip install noetic`) to use `NoeticLeibregBridge`."
            ) from exc

        bridge_cls = getattr(module, "PyFoldsBridge", None)
        if bridge_cls is None:
            raise ImportError(
                "Module `noetic.integration` was found but `PyFoldsBridge` is missing. "
                "Check your noetic version and integration package."
            )
        return bridge_cls(*args, **kwargs)


__all__ = ["NoeticLeibregBridge"]
