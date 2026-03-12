"""Factory para criação de instâncias FOLDSNet."""

from __future__ import annotations

from .model import FOLDSNet


_DATASETS = {
    "mnist": ((1, 28, 28), 10),
    "cifar10": ((3, 32, 32), 10),
    "cifar100": ((3, 32, 32), 100),
}


def create_foldsnet(variant: str, dataset: str) -> FOLDSNet:
    """Cria FOLDSNet com configuração padrão por dataset."""
    key = dataset.lower()
    if key not in _DATASETS:
        raise ValueError("Dataset inválido. Use 'mnist', 'cifar10' ou 'cifar100'.")

    input_shape, n_classes = _DATASETS[key]
    return FOLDSNet(input_shape=input_shape, n_classes=n_classes, variant=variant)
