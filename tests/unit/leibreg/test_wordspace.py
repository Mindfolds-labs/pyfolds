import pytest
import torch

from pyfolds.leibreg.wordspace import WordSpace


def test_import_and_initialization() -> None:
    ws = WordSpace(concept_count=32, dim_base=4, dim_context=2)
    assert ws.base_embedding.weight.shape == (32, 4)


def test_forward_normalization() -> None:
    ws = WordSpace(concept_count=32, dim_base=4)
    out = ws(torch.tensor([1, 2, 3], dtype=torch.long))
    assert out["q_total"].shape == (3, 4)
    assert torch.allclose(out["norm"], torch.ones_like(out["norm"]), atol=1e-5)


def test_distance_symmetry_and_similarity_monotonicity() -> None:
    ws = WordSpace(concept_count=16)
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    c = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
    dab = ws.distance(a, b)
    dba = ws.distance(b, a)
    assert torch.allclose(dab, dba)
    assert torch.all(ws.similarity(a, c) > ws.similarity(a, b))


def test_wave_rotation_preserves_norm() -> None:
    ws = WordSpace(concept_count=16, wave_enabled=True)
    q = torch.randn(2, 3, 4)
    qn = torch.nn.functional.normalize(q, p=2, dim=-1)
    rot = ws._apply_wave_rotation(qn, torch.tensor(0.7))
    assert torch.allclose(qn.norm(dim=-1), rot.norm(dim=-1), atol=1e-5)


def test_invalid_concept_ids() -> None:
    ws = WordSpace(concept_count=4)
    with pytest.raises(TypeError):
        ws(torch.tensor([0.1]))
    with pytest.raises(ValueError):
        ws(torch.tensor([5], dtype=torch.long))
