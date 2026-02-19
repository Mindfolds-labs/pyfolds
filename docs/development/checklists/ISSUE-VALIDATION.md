# Checklist de Conformidade — ISSUE para IA

Use este checklist antes de abrir PR com nova issue.

## Estrutura
- [ ] O arquivo segue `ISSUE-[NNN]-[slug].md`.
- [ ] O título inicia com `# ISSUE-[NNN]:`.
- [ ] Existe seção `## Metadados` com os 4 campos obrigatórios.
- [ ] Existe seção `## 1. Objetivo`.
- [ ] Existe seção `## 2. Escopo` com `### 2.1 Inclui:` e `### 2.2 Exclui:`.
- [ ] Existe seção `## 3. Artefatos Gerados` com tabela.
- [ ] Existe seção `## 4. Riscos` com tabela.
- [ ] Existe seção `## 5. Critérios de Aceite` com checkboxes.
- [ ] Existe seção `## 6. PROMPT:EXECUTAR`.
- [ ] O bloco `PROMPT:EXECUTAR` está em YAML (` ```yaml ... ``` `).

## Qualidade e rastreabilidade
- [ ] Todos os caminhos de artefato apontam para arquivos/pastas reais ou planejados.
- [ ] Dependências entre ISSUEs (campo `dependente`) são válidas.
- [ ] Links para ISSUEs citadas não estão quebrados.
- [ ] A fase da issue (`ativa`, `freeze` ou `legado`) está explícita e compatível com o workflow vigente.
- [ ] Mudanças estruturais (arquitetura, padrões de integração, contratos públicos) citam ADR aprovada.
- [ ] Existe referência explícita ao identificador da ADR aplicável (ex.: `ADR 0041`).
- [ ] A issue foi registrada em `docs/development/execution_queue.csv`.
- [ ] O HUB foi sincronizado (`python tools/sync_hub.py`).

## Regra de transição por fase
- [ ] **Fase ativa:** permite novas `ISSUE-*`.
- [ ] **Fase freeze:** permite nova issue apenas para correção crítica, com justificativa.
- [ ] **Fase legado:** bloqueia abertura de nova issue; uso somente para consulta histórica.

## Validação automatizada
- [ ] `python tools/validate_issue_format.py docs/development/prompts/relatorios/ISSUE-*.md`
- [ ] `python tools/check_issue_links.py docs/development/prompts/relatorios`
- [ ] `python tools/sync_hub.py --check`
