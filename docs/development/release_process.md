# Release Process (PyFolds)

## Objetivo
Padronizar o fluxo de release com rastreabilidade e checklist operacional auditável.

## Escopo
Aplica-se a releases de código, documentação e artefatos de benchmark.

## Papéis
- **Executor:** prepara versão, validações e evidências.
- **Maintainer:** revisa checklist e aprova publicação.

## Pré-condições
- Branch alvo atualizada e sem conflitos.
- CI verde (testes, docstrings públicas, links e HUB sync).
- `CHANGELOG.md` atualizado com impactos externos.

## Fluxo de release
1. **Planejamento**
   - Confirmar escopo (features/fixes/docs).
   - Confirmar ADRs relacionadas (quando aplicável).
2. **Hardening**
   - Rodar suíte local e checks obrigatórios.
   - Validar links de documentação.
3. **Versionamento**
   - Definir versão semântica.
   - Atualizar `CHANGELOG.md`.
4. **Publicação**
   - Criar tag de release.
   - Publicar artefatos (quando aplicável).
5. **Pós-release**
   - Registrar incidentes e ações corretivas.
   - Atualizar rastreabilidade no HUB/fila.

## Checklist auditável de release

### A. Validações técnicas
- [ ] `python -m compileall src/`
- [ ] `python tools/check_api_docs.py --strict`
- [ ] `python tools/check_links.py docs README.md`
- [ ] `python tools/sync_hub.py --check`
- [ ] `pytest tests/ -v`

### B. Artefatos e governança
- [ ] `CHANGELOG.md` atualizado.
- [ ] ADRs impactadas registradas/referenciadas.
- [ ] Fila `docs/development/execution_queue.csv` atualizada.
- [ ] HUB sincronizado (`docs/development/HUB_CONTROLE.md`).

### C. Publicação
- [ ] Versão semântica validada (tag).
- [ ] Release publicada com notas.
- [ ] Registro pós-release concluído.

## Evidências mínimas por release
- Hash do commit de release.
- Resultado dos comandos da seção A.
- Link da PR aprovada e tag publicada.
