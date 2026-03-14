from __future__ import annotations

import re

from train_mnist_folds import _default_run_id, _generate_short_id


def test_generate_short_id_uses_expected_charset_and_length() -> None:
    short_id = _generate_short_id(length=10)
    assert len(short_id) == 10
    assert re.fullmatch(r"[a-z0-9]{10}", short_id)


def test_default_run_id_contains_model_dataset_and_suffix() -> None:
    run_id = _default_run_id("foldsnet", "mnist")
    assert run_id.startswith("foldsnet_mnist_")
    suffix = run_id.removeprefix("foldsnet_mnist_")
    assert re.fullmatch(r"[a-z0-9]{6}", suffix)
