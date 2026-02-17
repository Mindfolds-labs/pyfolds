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
python tools/sync_hub.py --check
```

## Referências
- [Packaging](packaging.md)
