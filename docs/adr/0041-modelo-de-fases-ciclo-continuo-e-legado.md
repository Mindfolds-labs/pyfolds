# ADR 0041: Modelo de fases para ciclo contínuo e legado de ISSUE

- **Status:** Aceito
- **Data:** 2026-02-19
- **Decisores:** Maintainers PyFOLDS
- **Relacionados:** [ADR 0001](../governance/adr/legado/0001-import-contract-and-release-readiness.md), [ADR 0040](../governance/adr/legado/0040-conclusao-do-ciclo-issue-e-foco-em-execucao.md)

## Contexto

O ADR 0040 encerrou o ciclo ativo de abertura de novas `ISSUE-*` para reduzir overhead operacional.
Com a evolução do projeto, surgiu a necessidade de uma política mais flexível: permitir novas issues quando
houver planejamento ativo, restringir em períodos críticos e preservar histórico em modo legado.

Também foi identificado risco de mudanças estruturais sem lastro arquitetural explícito.
Para manter rastreabilidade e governança, essas mudanças devem referenciar ADR aprovada.

## Decisão

1. Instituir política de fases para governança de `ISSUE-*`:
   - **Fase ativa:** aceita novas issues com fluxo completo de criação, análise, execução e finalização.
   - **Fase freeze:** aceita novas issues somente para correções críticas.
   - **Fase legado:** mantém `ISSUE-*` apenas para consulta histórica, sem novas aberturas.
2. Exigir declaração explícita de fase nos templates e checklists de issue.
3. Exigir vínculo de ADR para mudanças estruturais (arquitetura, contratos públicos, padrões de integração).
4. Atualizar documentação operacional para substituir regra absoluta de bloqueio por regra contextual por fase.

## Consequências

### Positivas

- Reintroduz capacidade de planejamento formal quando necessário.
- Mantém controle de risco em janelas de freeze.
- Preserva o acervo legado sem perder governança.
- Aumenta rastreabilidade de decisões estruturais via ADR.

### Negativas

- Requer disciplina adicional para informar fase e ADR em cada nova issue.
- Pode aumentar esforço de revisão documental em períodos de alta demanda.

## Plano de implementação

- Atualizar `docs/development/prompts/README.md` removendo bloqueio absoluto de novas issues.
- Atualizar `docs/development/WORKFLOW_INTEGRADO.md` com política de fases.
- Atualizar `docs/development/checklists/ISSUE-VALIDATION.md` com regra de transição e referência ADR.
- Atualizar templates de relatórios para campos obrigatórios de fase e vínculo ADR.

## Critérios de aceite

- [x] Nova ADR registrada na sequência canônica (`0041`).
- [x] Política de fases documentada no workflow integrado.
- [x] Regra de validação com referência ADR aplicada ao checklist.
- [x] Templates atualizados com campos obrigatórios de fase e ADR.
