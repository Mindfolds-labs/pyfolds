"""Fuzz mínimo para entrada inválida no leitor `.fold`.

Casos extensivos e propriedades estão em `tests/unit/serialization/test_foldio.py`.
"""

import os

import pytest

from pyfolds.serialization import FoldReader


def test_fold_reader_rejects_random_payload(tmp_path):
    target = tmp_path / "random.fold"
    target.write_bytes(os.urandom(128))

    with pytest.raises((ValueError, RuntimeError, EOFError)):
        with FoldReader(str(target), use_mmap=False):
            pass
