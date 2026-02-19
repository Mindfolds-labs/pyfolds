# ISSUE-006 — Links quebrados em documentação interna

## Contexto
Durante a auditoria funcional foi executado:

```bash
python tools/validate_docs_links.py
```

O validador encontrou referências locais inválidas em múltiplos documentos.

## Evidência
Links reportados como quebrados:

- `docs/adr/0041-modelo-de-fases-ciclo-continuo-e-legado.md: ./0001-import-contract-and-release-readiness.md`
- `docs/adr/0041-modelo-de-fases-ciclo-continuo-e-legado.md: ./0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md`
- `docs/architecture/README.md: ../adr/0001-import-contract-and-release-readiness.md`
- `docs/development/HUB_CONTROLE.md: generated/dashboard.html`
- `docs/development/prompts/README.md: ../../adr/0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md`

## Impacto
- Navegação quebrada entre artefatos de governança.
- Redução da confiabilidade do processo de auditoria documental.
- Risco de perda de contexto histórico em ADRs.

## Critérios de aceite
- `python tools/validate_docs_links.py` retorna sucesso sem links quebrados.
- Todas as rotas locais apontam para arquivos existentes.
- Documentos de governança críticos (ADR/HUB/prompts) ficam navegáveis.
