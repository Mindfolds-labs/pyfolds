# Testing

## Objetivo
Descrever validações mínimas para mudanças no projeto.

## Escopo
Testes unitários/integrados e verificações de documentação.

## Definições/Termos
- **Smoke test:** validação rápida de funcionamento.

## Perfis de execução

### 0) Preflight sem Torch (tools/docs)
Use quando o ambiente não possui PyTorch e o objetivo é validar tooling/documentação.

```bash
PYTHONPATH=src pytest tests/tools/ -q
python tools/check_links.py docs/ README.md
python tools/validate_docs_links.py
```

### 1) Core-only (rápido)
Use quando alterar API pública/imports, validações básicas e contratos centrais.

```bash
PYTHONPATH=src pytest tests/unit/test_public_import_surface.py tests/unit/core/ -q
PYTHONPATH=src pytest tests/unit/test_backend_contracts.py -q -m "not tf"
```

### 2) Torch-full (unit + integração)
Perfil recomendado para mudanças no backend torch, estado interno e fluxo temporal.

```bash
PYTHONPATH=src pytest tests/unit/ tests/integration/ -q -m "not tf"
PYTHONPATH=src pytest tests/integration/test_temporal_sequence_stability.py -q -m "not tf"
```

### 3) TF-enabled (condicional)
Executar apenas em ambientes com TensorFlow instalado.

```bash
PYTHONPATH=src pytest tests/unit/test_backend_contracts.py -q -m tf
PYTHONPATH=src pytest tests/integration/test_temporal_sequence_stability.py -q -m tf
```

## Verificações de documentação
```bash
python tools/check_links.py docs/ README.md
python tools/docs_hub_audit.py --check
python tools/sync_hub.py --check
PYTHONPATH=src sphinx-build -W -b html docs docs/_build/html
```

## Referências
- [Packaging](packaging.md)
