import torch

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge
from pyfolds.leibreg.imagination import Imagination
from pyfolds.leibreg.reg_core import REGCore
from pyfolds.leibreg.wordspace import WordSpace


class _MemoryStub:
    def retrieve(self, query):
        _ = query
        return torch.ones(2, 6)


def _build_bridge() -> NoeticLeibregBridge:
    ws = WordSpace(text_dim=8, image_dim=10, memory_dim=6)
    return NoeticLeibregBridge(wordspace=ws, reg_core=REGCore(), imagination=Imagination())


def test_bridge_text_only_flow() -> None:
    bridge = _build_bridge()
    out = bridge(text_features=torch.randn(2, 8))
    assert out["text_point"] is not None
    assert out["image_point"] is None
    assert out["fused_point"] is not None and out["fused_point"].shape == (2, 4)


def test_bridge_image_only_flow() -> None:
    bridge = _build_bridge()
    out = bridge(image_features=torch.randn(2, 10))
    assert out["text_point"] is None
    assert out["image_point"] is not None
    assert out["fused_point"] is not None and out["fused_point"].shape == (2, 4)


def test_bridge_multimodal_flow() -> None:
    bridge = _build_bridge()
    out = bridge(text_features=torch.randn(2, 8), image_features=torch.randn(2, 10))
    assert out["text_point"] is not None
    assert out["image_point"] is not None
    assert out["reg_output"] is not None and out["reg_output"].shape == (2, 4)


def test_bridge_with_memory_flow() -> None:
    ws = WordSpace(text_dim=8, image_dim=10, memory_dim=6)
    bridge = NoeticLeibregBridge(wordspace=ws, associative_memory=_MemoryStub())
    out = bridge(text_features=torch.randn(2, 8), memory_query={"q": "cat"})
    assert out["memory_point"] is not None
    assert out["memory_point"].shape == (2, 4)


def test_bridge_output_keys_and_shapes() -> None:
    bridge = _build_bridge()
    out = bridge(text_features=torch.randn(3, 8))
    expected_keys = {
        "text_point",
        "image_point",
        "memory_point",
        "fused_point",
        "reg_output",
        "imagination_output",
        "imagination_confidence",
    }
    assert expected_keys.issubset(out.keys())
    assert out["reg_output"] is not None and out["reg_output"].shape == (3, 4)
    assert out["imagination_output"] is not None and out["imagination_output"].shape == (3, 4)
