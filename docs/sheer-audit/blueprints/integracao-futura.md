# Blueprint: Integração futura do Sheer Audit no pyfolds

## Objetivo

Integrar o Sheer Audit ao fluxo de documentação do projeto para produzir, versionar e publicar evidências de arquitetura/código.

## Fases sugeridas

1. **Fase 0 — Baseline (concluída nesta entrega)**
   - Configuração inicial (`docs/sheer-audit/sheer.toml`).
   - Relatório de teste do Sheer Audit.
   - Inventário de arquivos Python em JSON.

2. **Fase 1 — Automação local**
   - Criar script `scripts/run_sheer_audit.sh` (futuro) para:
     - validar configuração;
     - executar scan;
     - atualizar relatórios em `docs/sheer-audit/data`;
     - gerar sumário Markdown para revisão humana.

3. **Fase 2 — CI**
   - Rodar Sheer Audit em pull requests.
   - Publicar artefatos de auditoria no workflow.
   - Criar gate básico (ex.: falhar em violações arquiteturais críticas).

4. **Fase 3 — Publicação em documentação**
   - Incorporar resultados na documentação oficial.
   - Exibir evolução por release (delta de achados).

## Contrato de artefatos (proposto)

- `docs/sheer-audit/data/scan_summary.json`
- `docs/sheer-audit/data/default_python_files.json`
- `docs/sheer-audit/data/no_tests_python_files.json`
- `docs/sheer-audit/data/src_only_python_files.json`
- `docs/sheer-audit/data/code_map.json`
- `docs/sheer-audit/data/code_map.md`
- `docs/sheer-audit/data/repo_model.json`
- `docs/sheer-audit/data/uml/package.puml`
- `docs/sheer-audit/data/uml/class_overview.puml`
- `docs/sheer-audit/reports/sheer_audit_pytest.txt`
- `docs/sheer-audit/reports/sheer_execution_matrix.md`

## Checklist de integração

- [ ] Validar versão da CLI completa do Sheer Audit no ambiente.
- [ ] Conectar comandos oficiais `scan/report/uml/trace` quando disponíveis.
- [ ] Definir regras de arquitetura em `[architecture]` no `sheer.toml`.
- [ ] Publicar seção fixa "Auditoria de Código" na documentação.
