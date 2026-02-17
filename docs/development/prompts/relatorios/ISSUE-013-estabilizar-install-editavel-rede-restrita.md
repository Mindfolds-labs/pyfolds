# ISSUE-013: Estabilizar instalação editável em rede restrita + consolidar falhas da auditoria ADR-035

## Metadados

| Campo | Valor |
|---|---|
| Data | 2026-02-17 |
| Autor | Codex |
| Tipo | CODE |
| Prioridade | Crítica |

## 1. Objetivo
Executar uma rodada crítica da fase `AUDITORIA_SRC_TESTES_ADR35`, consolidando falhas com evidência reproduzível e abrindo a ISSUE-013 com detalhamento técnico dos achados.

## 2. Escopo

### 2.1 Inclui:
- Coleta de ambiente e baseline de dependências.
- Compilação de `src/`.
- Smoke import do pacote.
- Teste de instalação editável (`pip install -e .`).
- Validações de documentação e HUB.
- Execução da suíte de testes.
- Consolidação de achados por severidade (P0/P1/P2).
- Atualização da fila de execução.

### 2.2 Exclui:
- Correções estruturais no código-fonte.
- Mudanças de API pública.
- Alterações de comportamento sem evidência de log.

## 3. Artefatos Gerados
- `docs/development/prompts/relatorios/ISSUE-013-estabilizar-install-editavel-rede-restrita.md`
- `docs/development/prompts/logs/ISSUE-013/env.txt`
- `docs/development/prompts/logs/ISSUE-013/pip_list.txt`
- `docs/development/prompts/logs/ISSUE-013/compileall.txt`
- `docs/development/prompts/logs/ISSUE-013/imports_smoke.txt`
- `docs/development/prompts/logs/ISSUE-013/packaging_install.txt`
- `docs/development/prompts/logs/ISSUE-013/check_api_docs.txt`
- `docs/development/prompts/logs/ISSUE-013/check_links.txt`
- `docs/development/prompts/logs/ISSUE-013/validate_docs_links.txt`
- `docs/development/prompts/logs/ISSUE-013/sync_hub_check.txt`
- `docs/development/prompts/logs/ISSUE-013/pytest.txt`

## 4. Riscos
- Build isolation de `pip` depender de acesso externo para `setuptools>=61.0`.
- Ambientes restritos/proxy gerarem falso negativo de empacotamento.
- Warnings persistirem como dívida técnica e mascararem novos problemas.

## 5. Critérios de Aceite
- Logs completos em `docs/development/prompts/logs/ISSUE-013/`.
- Falhas detalhadas com comando reprodutível, causa provável e sugestão.
- Registro da ISSUE-013 em `execution_queue.csv`.

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
1) **Falha no `pip install -e .` com build isolation em ambiente com proxy restrito**
- Comando: `pip install -e .`
- Erro: `Could not find a version that satisfies the requirement setuptools>=61.0` após tentativas com `ProxyError` (`403 Forbidden`).
- Arquivos relacionados: `pyproject.toml`.
- Causa provável: build isolation exige download de dependências de build e o ambiente não alcança o índice externo.
- Sugestão: documentar e suportar oficialmente fluxo `--no-build-isolation`/mirror interno para ambientes restritos.
- Evidência: `docs/development/prompts/logs/ISSUE-013/packaging_install.txt`.

### P1 — Falhas funcionais
1) **Sem falhas funcionais da suíte principal**
- Comando: `PYTHONPATH=src pytest tests/ -v`
- Resultado: `198 passed, 3 deselected, 2 warnings`.
- Evidência: `docs/development/prompts/logs/ISSUE-013/pytest.txt`.

### P2 — Qualidade / manutenção
1) **Marker `performance` não registrado no pytest**
- Warning: `PytestUnknownMarkWarning: Unknown pytest.mark.performance`.
- Arquivos: `tests/performance/test_batch_speed.py`, `pytest.ini`.
- Sugestão: registrar marker em `pytest.ini`.
- Evidência: `docs/development/prompts/logs/ISSUE-013/pytest.txt`.

2) **RuntimeWarning esperado em cleanup de mmap**
- Warning: `RuntimeWarning: Erro ao fechar mmap ... forced close error`.
- Arquivo: `tests/unit/serialization/test_foldio.py`.
- Sugestão: capturar/asserir warning explicitamente para reduzir ruído operacional.
- Evidência: `docs/development/prompts/logs/ISSUE-013/pytest.txt`.

## EVIDÊNCIAS (logs)
- `docs/development/prompts/logs/ISSUE-013/env.txt`
- `docs/development/prompts/logs/ISSUE-013/pip_list.txt`
- `docs/development/prompts/logs/ISSUE-013/compileall.txt`
- `docs/development/prompts/logs/ISSUE-013/imports_smoke.txt`
- `docs/development/prompts/logs/ISSUE-013/packaging_install.txt`
- `docs/development/prompts/logs/ISSUE-013/check_api_docs.txt`
- `docs/development/prompts/logs/ISSUE-013/check_links.txt`
- `docs/development/prompts/logs/ISSUE-013/validate_docs_links.txt`
- `docs/development/prompts/logs/ISSUE-013/sync_hub_check.txt`
- `docs/development/prompts/logs/ISSUE-013/pytest.txt`

## 7. Revisão de Engenharia de Interação (UX) e padrão IEEE

### 7.1 Diagnóstico UX documental
- O fluxo técnico está completo, mas faltava um guia explícito de leitura/validação para reduzir ambiguidade em revisões humanas.
- A evidência estava correta, porém sem checklist unificado de qualidade de interação documental.

### 7.2 Melhorias aplicadas
- Criação do guia: `docs/development/guides/DOC-UX-IEEE-REVIEW.md`.
- Padronização de critérios de revisão para navegação, legibilidade, fechamento e rastreabilidade.
- Inclusão de checklist operacional alinhado a IEEE 828, IEEE 730 e ISO/IEC 12207.

### 7.3 Lacunas remanescentes
- `tools/check_issue_links.py` ainda reporta referências antigas inválidas em relatórios legados (fora do escopo desta ISSUE).
- Recomenda-se tratar essas pendências em issue dedicada de saneamento documental.

### 7.4 Recomendação prática
- Adotar o guia DOC-UX-IEEE como gate leve de revisão antes de abrir PRs de governança/documentação.
