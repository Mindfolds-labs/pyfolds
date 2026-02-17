# LOG — ISSUE-010-ESPECIAL corrigir estrutura docs

## Data
2026-02-17

## Etapas executadas
1. Inventário de arquivos soltos na raiz `docs/`.
2. Confirmação das pastas-alvo já existentes.
3. Remanejamento dos 8 arquivos soltos com `git mv`.
4. Verificação dos 6 órfãos solicitados (raiz limpa).
5. Atualização das referências/links para os novos caminhos.
6. Validação de links e consistência do HUB.

## Evidências (comandos)
```bash
git mv docs/ALGORITHM.md docs/science/ALGORITHM.md
git mv docs/SCIENTIFIC_LOGIC.md docs/science/SCIENTIFIC_LOGIC.md
git mv docs/METHODOLOGY.md docs/science/METHODOLOGY.md
git mv docs/TEST_PROTOCOL.md docs/science/TEST_PROTOCOL.md
git mv docs/API_REFERENCE.md docs/api/API_REFERENCE.md
git mv docs/installation.md docs/_quickstart/installation.md
git mv docs/quickstart.md docs/_quickstart/quickstart.md
git mv docs/BENCHMARKS.md docs/assets/BENCHMARKS.md
python tools/check_links.py
python tools/sync_hub.py
python tools/sync_hub.py --check
```

## Resultado
- Estrutura `docs/` normalizada conforme solicitado.
- Soltos removidos da raiz por remanejamento para subpastas corretas.
- Órfãos listados continuam ausentes da raiz.
- Links validados após alteração.

## Status final
✅ Concluída
