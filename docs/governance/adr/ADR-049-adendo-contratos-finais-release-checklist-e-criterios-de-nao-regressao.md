# ADR-049 — Adendo de contratos finais, checklist de release e critérios de não-regressão

## Status
Ativo

## Data
2026-03-11

## Contexto
Após revisar os ADRs vigentes, com foco em:

- `ADR-041` (ordem de execução de mixins, `theta_eff`, contrato de saída de layer);
- `docs/adr/ADR-005-sleep-replay.md` (replay offline com consolidação opcional de pruning);

ficou claro que os contratos já estavam tecnicamente definidos, porém distribuídos entre documentos com níveis diferentes de detalhe operacional.

Para evitar decisões redundantes e reduzir ambiguidade na fase de release, este ADR registra **somente decisões novas de governança operacional** (sem substituir ADR-041 nem ADR-005).

## Decisão
1. **Consolidar contrato final de execução por trilha**
   - As trilhas canônicas de execução passam a ser explicitamente tratadas como: `INFERENCE`, `ONLINE`, `BATCH`, `SLEEP`.
   - `theta_eff` permanece o critério de disparo quando disponível (com `theta` como estado/base observável).
   - `MPJRDLayer.forward` mantém `u_values` como saída primária, com alias legado `u` durante janela de compatibilidade.
   - Em `SLEEP`, replay e consolidação estrutural continuam desacoplados via `consolidate_pruning_after_replay`.

2. **Checklist de release obrigatório por trilha**
   - Toda release deve registrar evidência de validação para cada trilha (`INFERENCE`, `ONLINE`, `BATCH`, `SLEEP`) com comando executado e resultado.
   - A ausência de evidência em qualquer trilha bloqueia publicação.

3. **Critérios de não-regressão mínimos para aprovação**
   - **Contrato de saída**: leitores novos e legados devem funcionar (`u_values` e alias `u`).
   - **Contrato de threshold**: comportamento consistente de disparo com `theta_eff` nos caminhos com mixins.
   - **Batch/STDP**: manter invariância de escala relativa a tamanho de lote (normalização por média no eixo do batch).
   - **Sono/replay**: replay não deve forçar pruning sem flag explícita.
   - **Rastreabilidade**: relatório de release deve mapear cada verificação para a trilha correspondente.

## Consequências
### Positivas
- Governança de release mais objetiva e auditável.
- Menor chance de regressões silenciosas entre contratos canônicos e legados.
- Menor dispersão documental entre ADR e guias de consumo público.

### Negativas
- Aumento de disciplina operacional para fechar releases.
- Possível ampliação inicial do tempo de validação por exigir cobertura por trilha.

## Relação com ADRs anteriores
- **Complementa** `ADR-041`, sem alterar sua semântica de ordem e contrato.
- **Complementa** `docs/adr/ADR-005-sleep-replay.md`, preservando o desacoplamento entre replay e consolidação de pruning.

## Rollback
Se o checklist por trilha introduzir bloqueio operacional indevido, aplicar exceção temporária via registro explícito no relatório de release, com:
1. justificativa técnica,
2. prazo para regularização,
3. issue de acompanhamento.

