.PHONY: install install-cuda install-dev test test-install clean build publish

PYTHON ?= python
PIP ?= pip

# -------------------------
# Instalação normal (CPU)
# -------------------------
install:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e .

# -------------------------
# Instalação com CUDA 11.8
# -------------------------
install-cuda:
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu118
	$(PIP) install -e .

# -------------------------
# Instalação para desenvolvimento
# -------------------------
install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

# -------------------------
# Testes unitários
# -------------------------
test:
	pytest tests/ -v --cov=pyfolds --cov-report=term-missing

# -------------------------
# TESTE DE INSTALAÇÃO LIMPA (venv local)
# -------------------------
test-install: build
	$(PYTHON) -m venv venv_test_pyfolds
	venv_test_pyfolds\Scripts\python -m pip install --upgrade pip
	venv_test_pyfolds\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
	venv_test_pyfolds\Scripts\pip install dist\*.whl
	venv_test_pyfolds\Scripts\python -c "import pyfolds; print('OK:', pyfolds.__file__)"
	venv_test_pyfolds\Scripts\python test_install.py

# -------------------------
# Build para PyPI
# -------------------------
build:
	$(PYTHON) -m build
	@echo ✅ Wheel criado em dist/

# -------------------------
# Publicar no PyPI
# -------------------------
publish: build
	twine upload dist/*
	@echo ✅ Publicado no PyPI!

# -------------------------
# Limpeza (PowerShell)
# -------------------------
clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force build,dist,*.egg-info,.pytest_cache -ErrorAction SilentlyContinue"
	powershell -NoProfile -Command "Remove-Item -Force .coverage -ErrorAction SilentlyContinue"
	powershell -NoProfile -Command "Remove-Item -Recurse -Force venv_test_pyfolds -ErrorAction SilentlyContinue"
	powershell -NoProfile -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue"
	powershell -NoProfile -Command "Get-ChildItem -Recurse -File -Filter *.pyc | Remove-Item -Force -ErrorAction SilentlyContinue"
