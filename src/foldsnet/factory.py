"""Factory para criação de instâncias FOLDSNet."""

from __future__ import annotations

from .model import FOLDSNet


_DATASET_MAP = {
    "mnist": ((1, 28, 28), 10),
    "cifar10": ((3, 32, 32), 10),
    "cifar100": ((3, 32, 32), 100),
}


def create_foldsnet(variant: str, dataset: str) -> FOLDSNet:
    """Cria modelo FOLDSNet a partir de variante e dataset."""
    key = dataset.lower()
    if key not in _DATASET_MAP:
        raise ValueError(f"Dataset inválido: {dataset}")

    input_shape, n_classes = _DATASET_MAP[key]
    return FOLDSNet(input_shape=input_shape, n_classes=n_classes, variant=variant)
