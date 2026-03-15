import pytest
import torch

from foldsnet.model import FOLDSNet


def test_forward_mnist():
    model = FOLDSNet(input_shape=(1, 28, 28), n_classes=10, variant="4L")
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, 10), f"Esperado (4, 10), got {y.shape}"


def test_forward_cifar():
    model = FOLDSNet(input_shape=(3, 32, 32), n_classes=10, variant="4L")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)


def test_forward_5L():
    model = FOLDSNet(input_shape=(3, 32, 32), n_classes=100, variant="5L")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 100)


def test_forward_return_intermediates_shapes():
    model = FOLDSNet(input_shape=(1, 28, 28), n_classes=10, variant="2L")
    x = torch.randn(3, 1, 28, 28)
    out = model(x, return_intermediates=True)

    assert out["retina"].shape == (3, model.n_retina)
    assert out["lgn"].shape == (3, model.n_lgn)
    assert out["v1"].shape == (3, model.n_v1)
    assert out["it"].shape == (3, model.n_it)
    assert out["logits"].shape == (3, 10)


def test_forward_shape_mismatch_raises_clear_error():
    model = FOLDSNet(input_shape=(1, 28, 28), n_classes=10, variant="4L")
    with pytest.raises(ValueError, match="Shape de entrada incompatível"):
        model(torch.randn(2, 3, 32, 32))


def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match="Variante inválida"):
        FOLDSNet(input_shape=(1, 28, 28), n_classes=10, variant="7L")
