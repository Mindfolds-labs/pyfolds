#!/bin/bash
set -euo pipefail

echo "🚀 Iniciando Blindagem do PyFolds..."

# 1. Instala dependências faltantes
pip install cryptography pytest-cov -q

# 2. Roda os testes com auditoria de cobertura
PYTHONPATH=src pytest tests/ -v \
    --cov=pyfolds \
    --cov-report=term-missing \
    --cov-report=html:docs/coverage_report

# 3. Verifica se o teste de criptografia (ECC) passou
PYTHONPATH=src pytest tests/unit/serialization/test_foldio.py::test_fold_signature_roundtrip_if_cryptography_available -v

echo "✅ Processo concluído. Verifique docs/coverage_report/index.html"
