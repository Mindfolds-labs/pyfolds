import pytest
import torch

from pyfolds.leibreg.wordspace import WordSpace


def test_project_text_shape() -> None:
    ws = WordSpace(text_dim=8, image_dim=16, memory_dim=12)
    x = torch.randn(3, 8)
    out = ws.project_text(x)
    assert out.shape == (3, 4)


def test_project_image_shape() -> None:
    ws = WordSpace(text_dim=8, image_dim=16, memory_dim=12)
    x = torch.randn(2, 5, 16)
    out = ws.project_image(x)
    assert out.shape == (2, 5, 4)


def test_project_invalid_modality() -> None:
    ws = WordSpace(text_dim=8, image_dim=16, memory_dim=12)
    with pytest.raises(ValueError, match="Modalidade inválida"):
        ws.project(torch.randn(1, 8), modality="audio")


def test_project_batch_consistency() -> None:
    ws = WordSpace(text_dim=8, image_dim=16, memory_dim=12, normalize=False)
    batch = torch.randn(4, 8)
    out1 = ws.project_text(batch)
    out2 = torch.stack([ws.project_text(batch[i : i + 1]).squeeze(0) for i in range(batch.shape[0])], dim=0)
    assert torch.allclose(out1, out2, atol=1e-6)
