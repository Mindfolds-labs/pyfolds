# Testing

## Objetivo
Descrever validações mínimas para mudanças no projeto.

## Escopo
Testes unitários/integrados e verificações de documentação.

## Definições/Termos
- **Smoke test:** validação rápida de funcionamento.

## Conteúdo técnico
```bash
PYTHONPATH=src pytest tests/ -q
python tools/check_links.py docs/ README.md
python tools/docs_hub_audit.py --check
python tools/sync_hub.py --check
PYTHONPATH=src sphinx-build -W -b html docs docs/_build/html
```

## Referências
- [Packaging](packaging.md)
