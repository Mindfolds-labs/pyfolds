import torch
import pytest

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


def test_bridge_end_to_end_import() -> None:
    pytest.importorskip("noetic.integration")
    from pyfolds.leibreg import NoeticLeibregBridge

    assert callable(NoeticLeibregBridge)
