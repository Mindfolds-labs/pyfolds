import torch

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge


class _AssocMemory:
    def retrieve(self, query):
        _ = query
        return [1, 2]


def test_bridge_partial_modality_text_only() -> None:
    bridge = NoeticLeibregBridge(dim_text=8, dim_concept=4)
    out = bridge(text_features=torch.randn(3, 8))
    assert out["concept_point"].shape == (3, 4)


def test_bridge_output_contract() -> None:
    bridge = NoeticLeibregBridge(dim_text=8, dim_image=10, dim_concept=4, associative_memory=_AssocMemory())
    out = bridge(text_features=torch.randn(2, 8), image_features=torch.randn(2, 10), memory_query={"q": "x"})
    assert {"concept_point", "activated_concepts", "memory_hits", "memory", "wave_metrics"}.issubset(out.keys())


def test_bridge_concept_ids_with_wave_source() -> None:
    class _Osc:
        def __init__(self) -> None:
            self.phase = torch.tensor(0.4)

    class _Wave:
        oscillators = [_Osc(), _Osc()]

    bridge = NoeticLeibregBridge(dim_text=8, dim_concept=4, concept_count=32)
    out = bridge(text_features=torch.randn(2, 8), concept_ids=torch.tensor([1, 2]), wave_source=_Wave())
    assert out["concept_point"].shape[-1] == 4
