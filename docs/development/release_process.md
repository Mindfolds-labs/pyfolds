# Release Process (PyFolds)

## Objetivo
Padronizar o fluxo de release com rastreabilidade e checklist operacional.

## Escopo
Aplica-se a releases de código, documentação e artefatos de benchmark.

## Pré-condições
- Branch alvo atualizada.
- CI verde (testes, validações de docs e HUB sync).
- Changelog atualizado para impactos externos.

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
   - Registrar eventuais incidentes e ações corretivas.

## Checklist de release
- [ ] Versionamento semântico validado.
- [ ] `CHANGELOG.md` atualizado.
- [ ] Testes automatizados verdes.
- [ ] `python tools/sync_hub.py --check` verde.
- [ ] Links de docs validados.
- [ ] Tag criada e release publicada.
