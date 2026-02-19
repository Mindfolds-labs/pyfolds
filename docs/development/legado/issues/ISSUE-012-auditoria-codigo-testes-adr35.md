# ISSUE-012: Auditoria src, testes e ADR-035

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex |
| Tipo | CODE |
| Prioridade | Crítica |

## 1. Objetivo
Executar auditoria do repositório com foco em `src/`, validar importação/instalação e testes com evidências em logs, e formalizar a decisão de priorização em ADR-035.

## 2. Escopo

### 2.1 Inclui:
- Auditoria de importação e carregamento de módulos em `src/`.
- Validação prática de instalação editável e import do pacote principal.
- Execução de validações: `compileall`, `check_api_docs`, `check_links`, `sync_hub --check`.
- Execução da suíte: `PYTHONPATH=src pytest tests/ -v`.
- Consolidação de achados priorizados (P0/P1/P2) com evidências.
- Criação do ADR-035 e atualização do índice de ADR.

### 2.2 Exclui:
- Implementar correções grandes em código-fonte.
- Alterar API pública.
- Melhorias sem evidência empírica nos logs.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-012-auditoria-codigo-testes-adr35.md`
- `docs/development/prompts/logs/ISSUE-012/env.txt`
- `docs/development/prompts/logs/ISSUE-012/pip_list.txt`
- `docs/development/prompts/logs/ISSUE-012/compileall.txt`
- `docs/development/prompts/logs/ISSUE-012/imports_smoke.txt`
- `docs/development/prompts/logs/ISSUE-012/packaging_install.txt`
- `docs/development/prompts/logs/ISSUE-012/check_api_docs.txt`
- `docs/development/prompts/logs/ISSUE-012/check_links.txt`
- `docs/development/prompts/logs/ISSUE-012/sync_hub_check.txt`
- `docs/development/prompts/logs/ISSUE-012/pytest.txt`
- `docs/governance/adr/ADR-035-auditoria-src-testes.md`
- `docs/governance/adr/INDEX.md`
- `docs/development/execution_queue.csv`

## 4. Riscos
- Escopo ampliar para correção ampla em vez de auditoria.
- Dependência de rede/proxy para build isolation em `pip install -e .`.
- Warnings de teste mascararem qualidade sem falha funcional.

## 5. Critérios de Aceite
- Pasta `docs/development/prompts/logs/ISSUE-012/` criada com logs requeridos.
- Falhas e observações registradas com comando reprodutível e severidade.
- ADR-035 criado e indexado.
- `python tools/sync_hub.py --check` executado com sucesso.

## 6. PROMPT:EXECUTAR
```yaml
fase: AUDITORIA_SRC_TESTES_ADR35
prioridade: CRITICA
responsavel: CODEX

acoes:
  - coletar_ambiente
  - compilar_src
  - smoke_import
  - testar_packaging_editavel
  - rodar_validacoes_docs_hub
  - executar_testes
  - consolidar_achados
  - criar_adr_035
  - atualizar_fila_execucao
```

## ACHADOS

### P0 — Quebra execução / build / import
1) **Falha de instalação editável com build isolation em ambiente com proxy restrito**
- Comando: `pip install -e .`
- Erro: `ERROR: Could not find a version that satisfies the requirement setuptools>=61.0` após `ProxyError`.
- Arquivos: `pyproject.toml`
- Causa provável: ausência de acesso ao índice de pacotes para dependências de build no modo isolado.
- Sugestão de correção: documentar fluxo offline/mirror e suportar `pip install -e . --no-build-isolation` em ambientes restritos.
- Evidência: `docs/development/prompts/logs/ISSUE-012/packaging_install.txt`

### P1 — Falhas de teste / comportamento incorreto
1) **Sem falhas funcionais na suíte principal**
- Comando: `PYTHONPATH=src pytest tests/ -v`
- Resultado: `198 passed, 3 deselected, 2 warnings`.
- Arquivos: `tests/`
- Causa provável: N/A (sem falha funcional).
- Sugestão de correção: manter baseline; tratar warnings como dívida técnica separada.
- Evidência: `docs/development/prompts/logs/ISSUE-012/pytest.txt`

### P2 — Qualidade / manutenção (com evidência)
1) **Marcador `performance` não registrado no pytest**
- Comando: `PYTHONPATH=src pytest tests/ -v`
- Erro/Warning: `PytestUnknownMarkWarning: Unknown pytest.mark.performance`
- Arquivos: `tests/performance/test_batch_speed.py`, `pytest.ini`
- Causa provável: marker sem cadastro em configuração.
- Sugestão de correção: registrar marker em `pytest.ini`.
- Evidência: `docs/development/prompts/logs/ISSUE-012/pytest.txt`

2) **RuntimeWarning controlado em teste de cleanup de mmap**
- Comando: `PYTHONPATH=src pytest tests/ -v`
- Erro/Warning: `RuntimeWarning: Erro ao fechar mmap ... forced close error`
- Arquivos: `tests/unit/serialization/test_foldio.py`
- Causa provável: cenário intencional de erro forçado em cleanup.
- Sugestão de correção: capturar/asserir warning explicitamente para reduzir ruído.
- Evidência: `docs/development/prompts/logs/ISSUE-012/pytest.txt`

## EVIDÊNCIAS (logs)
- `docs/development/prompts/logs/ISSUE-012/env.txt`
- `docs/development/prompts/logs/ISSUE-012/pip_list.txt`
- `docs/development/prompts/logs/ISSUE-012/compileall.txt`
- `docs/development/prompts/logs/ISSUE-012/imports_smoke.txt`
- `docs/development/prompts/logs/ISSUE-012/packaging_install.txt`
- `docs/development/prompts/logs/ISSUE-012/pytest.txt`
- `docs/development/prompts/logs/ISSUE-012/sync_hub_check.txt`
- `docs/development/prompts/logs/ISSUE-012/check_api_docs.txt`
- `docs/development/prompts/logs/ISSUE-012/check_links.txt`

## RECOMENDAÇÕES / PRÓXIMAS ISSUES
- CODE: ISSUE-013 — Estabilizar instalação editável em cenários com rede restrita/proxy.
- TEST: ISSUE-014 — Registrar marker `performance` e tratar warnings esperados da suíte.
- DOCS: ISSUE-015 — Documentar troubleshooting de instalação (`build isolation`, mirror, offline).
