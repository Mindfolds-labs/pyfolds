# FOLD Specification (stub inicial)

Este documento define o ponto de entrada canônico da especificação do container `.fold` dentro do repositório.

## Escopo

- Formato do container `.fold`/`.mind`.
- Integridade (CRC32C, hash hierárquico) e proteção ECC.
- Leitura parcial por chunks e validações de segurança.

## Referências internas

- Implementação principal: `src/pyfolds/serialization/foldio.py`
- Testes de serialização: `tests/unit/serialization/test_foldio.py`
- Fundamentação teórica do formato: `docs/theory/FOLD_MIND_FORMAT.md`

## Próximos incrementos

1. Formalizar gramática binária dos headers/chunks.
2. Especificar matriz de compatibilidade por versão.
3. Documentar exemplos de corrupção e recuperação ECC.
