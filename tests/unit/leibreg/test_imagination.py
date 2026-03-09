import torch

from pyfolds.leibreg.imagination import Imagination


def test_imagination_preserves_shape_contract() -> None:
    model = Imagination(hidden_dim=12)
    x = torch.randn(2, 4)
    y, conf = model(x)
    assert y.shape == x.shape
    assert conf is not None and conf.shape == (2, 1)


def test_imagination_deterministic_in_eval_mode() -> None:
    model = Imagination(hidden_dim=12, dropout=0.4)
    model.eval()
    x = torch.randn(2, 4)
    y1, c1 = model(x)
    y2, c2 = model(x)
    assert torch.allclose(y1, y2)
    assert c1 is not None and c2 is not None and torch.allclose(c1, c2)


def test_imagination_confidence_bounds() -> None:
    model = Imagination(hidden_dim=8)
    x = torch.randn(4, 4)
    _, conf = model(x)
    assert conf is not None
    assert ((conf >= 0.0) & (conf <= 1.0)).all()
