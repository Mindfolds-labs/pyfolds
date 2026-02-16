# FOLD Quality Plan

Task|Status|Owner|Start|End|Ev|RB
---|---|---|---|---|---|---
A1|TODO|||||git restore .
A2|TODO|||||git restore src/pyfolds/serialization
A3|TODO|||||git restore tests
A4|TODO|||||git restore benchmarks .github/workflows

## Setup e anti-conflito
- `pip install -e ".[dev,serialization]" ruff bandit pytest-cov hypothesis pytest-benchmark`

## Split de escopo
- **A1 = src + root**
- **A2 = serialization + docs/spec**
- **A3 = tests**
- **A4 = benchmarks + .github**

## Validação de caminhos oficiais (repo)
- Fontes consultadas no repositório: `README.md` (portal de docs), `docs/README.md` (índice), estrutura atual de arquivos.
- Caminhos canônicos confirmados para este plano:
  - `docs/spec/FOLD_SPECIFICATION.md`
  - `tests/test_fold_corruption.py`
  - `tests/test_fold_fuzz.py`
  - `benchmarks/bench_foldio.py`
  - `.github/workflows/benchmarks.yml`

## A1
- **Objetivo:** lint / tipos / segurança
- **Branch sugerida:** `audit/src-structure`
- **Arquivo de controle:** `FOLD_QUALITY_PLAN.md`
- **Comandos:**
  - `ruff check src`
  - `mypy src`
  - `bandit -r src`
- **Critério de aceite:**
  - `ruff` e `mypy` retornam zero
  - `bandit` sem findings HIGH

## A2
- **Objetivo:** hardening de I/O
- **Branch sugerida:** `hardening/serialization`
- **Arquivo alvo:** `docs/spec/FOLD_SPECIFICATION.md`
- **Escopo adicional:** `src/pyfolds/serialization/foldio.py`
- **Comando:** `pytest -q`
- **Critério de aceite:**
  - limites de índice (`MAX_INDEX_SIZE`)
  - validação de bounds/EOF
  - uso de `torch.load(..., weights_only=True)`

## A3
- **Objetivo:** ampliar testes
- **Branch sugerida:** `quality/testing-suite`
- **Arquivos alvo:**
  - `tests/test_fold_corruption.py`
  - `tests/test_fold_fuzz.py`
- **Comando:** `pytest tests -q`
- **Critério de aceite:** fail-fast + no-crash

## A4
- **Objetivo:** benchmark + CI
- **Branch sugerida:** `perf/benchmark-ci`
- **Arquivos alvo:**
  - `benchmarks/bench_foldio.py`
  - `.github/workflows/benchmarks.yml`
- **Comando:**
  - `pytest benchmarks --benchmark-only --benchmark-json=benchmark.json`
- **Critério de aceite:** artifact `benchmark.json`
