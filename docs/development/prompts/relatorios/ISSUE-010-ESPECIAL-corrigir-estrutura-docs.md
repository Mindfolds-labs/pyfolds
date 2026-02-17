# ISSUE-010-ESPECIAL: Corrigir estrutura `docs/` — remover soltos e órfãos

## Metadados

| Campo | Valor |
|---|---|
| **Tipo** | GOVERNANCE |
| **Data** | 2026-02-17 |
| **Autor** | Codex |
| **Objetivo** | Normalizar `docs/` antes da ISSUE-011 sem perda de conteúdo |

## 1. Escopo executado

### Inclui
- Verificação de arquivos soltos em `docs/` versus subpastas-alvo.
- Remanejamento dos arquivos da raiz para as pastas corretas já existentes.
- Verificação dos 6 órfãos listados na solicitação.
- Revisão de links após remanejamento.
- Validação de links com `python tools/check_links.py`.
- Sincronização do HUB com `python tools/sync_hub.py --check`.

### Exclui (respeitado)
- Não alterar conteúdo técnico dos arquivos movidos.
- Não mexer em `src/`.
- Não criar novas pastas.

## 2. Remanejamento executado (8 soltos)

| Arquivo origem em `docs/` | Destino final | Situação |
|---|---|---|
| `ALGORITHM.md` | `docs/science/ALGORITHM.md` | Movido |
| `SCIENTIFIC_LOGIC.md` | `docs/science/SCIENTIFIC_LOGIC.md` | Movido |
| `METHODOLOGY.md` | `docs/science/METHODOLOGY.md` | Movido |
| `TEST_PROTOCOL.md` | `docs/science/TEST_PROTOCOL.md` | Movido |
| `API_REFERENCE.md` | `docs/api/API_REFERENCE.md` | Movido |
| `installation.md` | `docs/_quickstart/installation.md` | Movido |
| `quickstart.md` | `docs/_quickstart/quickstart.md` | Movido |
| `BENCHMARKS.md` | `docs/assets/BENCHMARKS.md` | Movido |

## 3. Verificação dos 6 órfãos

| Arquivo órfão listado | Encontrado na raiz `docs/`? | Situação |
|---|---|---|
| `ARCHITECTURE_V3.md` | Não | Já removido em ciclo anterior |
| `SCIENTIFIC_BASIS_V3.md` | Não | Já removido em ciclo anterior |
| `PHASE_CODING.md` | Não | Raiz limpa; documento existe em `docs/research/wave/` |
| `WAVELENGTH_MAPPING.md` | Não | Já removido em ciclo anterior |
| `COOPERATIVE_INTEGRATION.md` | Não | Já removido em ciclo anterior |
| `FOLD_SPECIFICATION.md` | Não | Raiz limpa; documento existe em `docs/architecture/specs/` |

## 4. Ajustes de referências

Foram atualizados links em índices e hubs para os novos destinos em:
- `README.md`
- `docs/index.md`
- `docs/science/README.md`
- `docs/_quickstart/README.md`
- `docs/architecture/README.md`
- `docs/specifications/README.md`
- `docs/DEVELOPMENT_HUB.md`
- `docs/research/README.md`
- `docs/architecture/blueprints/README.md`
- `docs/development/benchmarking.md`

## 5. Conclusão

- `docs/` raiz foi normalizada com remoção de soltos por remanejamento seguro.
- Nenhuma perda de conteúdo: operação feita via `git mv`.
- Estrutura pronta para continuação da ISSUE-011.

## 6. Comandos de validação

```bash
python tools/check_links.py
python tools/sync_hub.py
python tools/sync_hub.py --check
```
