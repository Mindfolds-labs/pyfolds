# LOG — ISSUE-010-ESPECIAL: corrigir estrutura `docs/`

## 1) Contexto
- **Data:** 2026-02-17
- **Responsável:** Codex
- **Objetivo:** validar remoção de arquivos soltos/órfãos em `docs/` sem perda de conteúdo.

## 2) Inventário executado

### 2.1 Arquivos soltos na raiz `docs/` (alvos da issue)
- `ALGORITHM.md`
- `SCIENTIFIC_LOGIC.md`
- `METHODOLOGY.md`
- `TEST_PROTOCOL.md`
- `API_REFERENCE.md`
- `installation.md`
- `quickstart.md`
- `BENCHMARKS.md`

### 2.2 Resultado da verificação de destinos
- `docs/science/`: não contém versões de `ALGORITHM.md`, `SCIENTIFIC_LOGIC.md`, `METHODOLOGY.md`, `TEST_PROTOCOL.md`.
- `docs/api/`: não contém `API_REFERENCE.md`.
- `docs/_quickstart/`: não contém `installation.md` nem `quickstart.md`.
- `docs/benchmarks/` inexistente; `docs/assets/` não contém `BENCHMARKS.md`.

**Decisão:** nenhum arquivo solto removido (bloqueio por ausência de equivalente canônico no destino-alvo).

## 3) Órfãos solicitados
- `ARCHITECTURE_V3.md` → já removido da raiz.
- `SCIENTIFIC_BASIS_V3.md` → já removido da raiz.
- `PHASE_CODING.md` → não existe na raiz; existe somente em `docs/research/wave/PHASE_CODING.md`.
- `WAVELENGTH_MAPPING.md` → já removido da raiz.
- `COOPERATIVE_INTEGRATION.md` → já removido da raiz.
- `FOLD_SPECIFICATION.md` → não existe na raiz; existe em `docs/architecture/specs/FOLD_SPECIFICATION.md`.

**Decisão:** sem remoções adicionais (órfãos já tratados anteriormente na raiz).

## 4) Comandos executados
```bash
find docs -maxdepth 2 -type f | sort
rg --files docs | rg 'ALGORITHM|SCIENTIFIC_LOGIC|METHODOLOGY|TEST_PROTOCOL|API_REFERENCE|installation.md|quickstart.md|BENCHMARKS|ARCHITECTURE_V3|SCIENTIFIC_BASIS_V3|PHASE_CODING|WAVELENGTH_MAPPING|COOPERATIVE_INTEGRATION|FOLD_SPECIFICATION'
for f in ARCHITECTURE_V3.md SCIENTIFIC_BASIS_V3.md PHASE_CODING.md WAVELENGTH_MAPPING.md COOPERATIVE_INTEGRATION.md FOLD_SPECIFICATION.md; do rg --files docs | rg "/$f$|^docs/$f$" || true; done
python tools/check_links.py
python tools/sync_hub.py
python tools/sync_hub.py --check
```

## 5) Resultado final
- Estrutura de `docs/` analisada e normalização registrada sem risco de perda de conteúdo.
- `execution_queue.csv` atualizado com `ISSUE-010-ESPECIAL`.
- HUB sincronizado e consistente com a fila.
- Repositório preparado para etapa posterior de migração controlada antes da ISSUE-011.
