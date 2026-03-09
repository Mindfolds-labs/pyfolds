import torch

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge
from pyfolds.leibreg import REGCore, SIGReg, WordSpace


def test_import_surface() -> None:
    assert WordSpace is not None
    assert REGCore is not None
    assert SIGReg is not None


def test_end_to_end_smoke() -> None:
    ws = WordSpace(concept_count=64, dim_base=4, text_dim=8)
    concept = ws(torch.tensor([1, 2, 3], dtype=torch.long))["q_total"]
    core = REGCore(dim=4, depth=2)
    y = core(concept.unsqueeze(0))
    loss = SIGReg(dim=4)(y)
    assert y.shape == (1, 3, 4)
    assert torch.isfinite(loss)


def test_bridge_end_to_end() -> None:
    bridge = NoeticLeibregBridge(dim_text=8, dim_image=8, dim_concept=4, concept_count=64)
    out = bridge(text_features=torch.randn(4, 8), image_features=torch.randn(4, 8), concept_ids=torch.tensor([1, 2, 3, 4]))
    assert out["concept_point"].shape == (4, 4)
