# 🧾 README — Relatórios de Issues

Este diretório contém relatórios no padrão de governança de prompts.

## ✅ Regras obrigatórias
- Todo novo `ISSUE-[NNN]-*.md` deve passar em `tools/validate_issue_format.py`.
- O conteúdo analítico deve usar `ISSUE-003-auditoria-completa.md` como referência canônica.
- Para cada ISSUE, deve existir um `EXEC-[NNN]-*.md` correspondente.
- Sempre que `execution_queue.csv` for atualizado, `python tools/sync_hub.py` deve ser executado e `HUB_CONTROLE.md` deve mudar no mesmo commit.

## 🧱 Estrutura obrigatória da ISSUE (validador)
1. `# ISSUE-NNN: ...`
2. `## Metadados`
3. `## 1. Objetivo`
4. `## 2. Escopo`
5. `### 2.1 Inclui:`
6. `### 2.2 Exclui:`
7. `## 3. Artefatos Gerados`
8. `## 4. Riscos`
9. `## 5. Critérios de Aceite`
10. `## 6. PROMPT:EXECUTAR` com bloco YAML

## 📚 Referências
- Modelo base: `ISSUE-000-template.md`
- Relatório canônico: `ISSUE-003-auditoria-completa.md`
