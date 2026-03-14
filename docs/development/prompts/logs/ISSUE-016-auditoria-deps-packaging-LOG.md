# LOG de execução — ISSUE-016 auditoria deps/packaging

## Comandos executados

1. `python -m compileall src/pyfolds` → sucesso.
2. `PYTHONPATH=src pytest tests/unit/test_public_import_surface.py -q` → 4 passed.
3. `python -m pip install --dry-run -e .` → sucesso.
4. `python -m pip install --dry-run -e '.[dev]'` → sucesso.
5. `python -m pip install --dry-run --ignore-installed --only-binary=:all: --python-version 3.8 .` → sucesso (simulado).
6. `python -m pip install --dry-run --ignore-installed --only-binary=:all: --python-version 3.9 .` → sucesso (simulado).
7. `PYENV_VERSION=3.10.19 python -m pip install --dry-run -e .` → sucesso.
8. `PYENV_VERSION=3.11.14 python -m pip install --dry-run -e .` → sucesso.

## Evidências de decisão
- `setup.cfg` reduzido para impedir drift frente ao `pyproject.toml`.
- `requirements.txt` realinhado para runtime canônico.
- Política de depreciação v1 atualizada para remover inconsistência com o comportamento real de `src/pyfolds/__init__.py`.
- Configuração pytest duplicada removida do `pyproject.toml`.
