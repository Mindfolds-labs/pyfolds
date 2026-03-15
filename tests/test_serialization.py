import pytest
import torch

from foldsnet.factory import create_foldsnet
from foldsnet.model import FOLDSNet


def test_save_load_fold(tmp_path):
    model = create_foldsnet("4L", "mnist")
    path = str(tmp_path / "test.fold")
    model.save(path, format="fold")
    loaded = FOLDSNet.load(path, format="fold")
    x = torch.randn(2, 1, 28, 28)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_save_load_mind(tmp_path):
    model = create_foldsnet("4L", "mnist")
    path = str(tmp_path / "test.mind")
    model.save(path, format="mind", include_metadata=True)
    loaded = FOLDSNet.load(path, format="mind")
    x = torch.randn(2, 1, 28, 28)
    assert torch.allclose(model(x), loaded(x), atol=1e-6)


def test_save_invalid_extension_rejected(tmp_path):
    model = create_foldsnet("4L", "mnist")
    with pytest.raises(ValueError, match="Extensão incompatível"):
        model.save(str(tmp_path / "bad.pt"), format="fold")
