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


def test_forward_5l():
    model = FOLDSNet(input_shape=(3, 32, 32), n_classes=100, variant="5L")
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 100)
