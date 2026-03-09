import torch

from pyfolds.advanced.engram import EngramBank
from pyfolds.leibreg import (
    Imagination,
    LeibnizLayer,
    NoeticLeibregBridge,
    ProximityAttention,
    REGCore,
    SIGReg,
    WordSpace,
)


def test_import_surface():
    assert WordSpace is not None
    assert REGCore is not None


def test_wordspace_shapes_and_symmetry():
    ws = WordSpace(concept_count=16, dim_base=4, dim_context=2)
    ids = torch.tensor([[1, 2, 3]])
    out = ws(ids)
    assert out["q_base"].shape == (1, 3, 4)
    assert out["q_total"].shape == (1, 3, 6)
    a = out["q_total"][:, 0, :]
    b = out["q_total"][:, 1, :]
    assert torch.allclose(ws.similarity(a, b), ws.similarity(b, a), atol=1e-6)


def test_wordspace_rejects_invalid_ids_and_empty():
    ws = WordSpace(concept_count=4)
    try:
        ws(torch.tensor([], dtype=torch.long))
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        ws(torch.tensor([9], dtype=torch.long))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_leibniz_layer_gradient_flow():
    layer = LeibnizLayer(dim_input=8, dim_output=4, normalize_output=True)
    x = torch.randn(2, 5, 8, requires_grad=True)
    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_proximity_attention_normalization_and_mask():
    attn = ProximityAttention(dim=4, kernel="gaussian", temperature=0.7)
    x = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    y = attn(x, mask=mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_regcore_and_sigreg_small_inputs():
    core = REGCore(dim=4, depth=2)
    x = torch.zeros(1, 1, 4, requires_grad=True)
    y = core(x)
    reg = SIGReg(weight=0.1)(y)
    (y.mean() + reg).backward()
    assert x.grad is not None


def test_wave_modulation_path():
    ws = WordSpace(concept_count=8, wave_enabled=True)
    ids = torch.tensor([0, 1, 2], dtype=torch.long)
    wave_phase = torch.tensor(0.5)
    out = ws(ids, wave_phase=wave_phase)
    assert out["q_total"].shape == (3, 4)


def test_imagination_with_engram_bank_mocked_data():
    bank = EngramBank(max_engrams=32, n_frequencies=4)
    bank.create_engram(torch.tensor([1.0, 0.0, 0.0, 0.0]), "alpha", age=0.0, phase=0.0, meridiem="am")
    bank.create_engram(torch.tensor([0.0, 1.0, 0.0, 0.0]), "beta", age=0.0, phase=0.0, meridiem="am")
    imag = Imagination(engram_bank=bank)
    out = imag(torch.tensor([1.0, 0.0, 0.0, 0.0]), k=1)
    assert len(out["concepts"]) == 1
    assert out["backend"] == "engram_bank"


def test_noetic_bridge_smoke_end_to_end():
    bank = EngramBank(max_engrams=16, n_frequencies=4)
    bank.create_engram(torch.tensor([1.0, 0.0, 0.0, 0.0]), "alpha", age=0.0, phase=0.0, meridiem="am")
    bridge = NoeticLeibregBridge(dim_text=6, dim_concept=4, concept_count=32, engram_bank=bank)
    texto = torch.randn(5, 6)
    concept_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    out = bridge(texto=texto, concept_ids=concept_ids)
    assert out["concept_point"].shape[-1] == 4
    assert "memory" in out
