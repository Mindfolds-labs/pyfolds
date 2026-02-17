# ADR-037 — Análise integral de execução (ISSUE-025) e atualização contínua de benchmark

## Status
Aceito

## Contexto
Foi solicitada validação integral do repositório com execução direta dos testes, identificação de erros reais (vs. suposições), geração de relatório técnico e atualização dos resultados de benchmark registrados em documentação.

Também foi apontada necessidade de explicitar a governança do ciclo "analisar → executar → relatar → aprovar" para evitar lacunas entre execução técnica e atualização de artefatos (`docs/assets/benchmarks_results.json` e `docs/assets/BENCHMARKS.md`).

## Decisão
Instituir o fluxo mínimo obrigatório para demandas operacionais equivalentes à ISSUE-025:
1. validar compilação (`python -m compileall src`);
2. validar suíte principal (`PYTHONPATH=src pytest -q`);
3. atualizar benchmark com script oficial;
4. regenerar relatório de benchmark em Markdown;
5. registrar ISSUE/EXEC na trilha de governança;
6. sincronizar HUB e manter ADR de decisão consolidada.

## Alternativas consideradas
1. **Atualizar apenas benchmark sem regressão de testes**
   - Prós: menor custo de execução.
   - Contras: reduz confiança nos números e no estado funcional do código.

2. **Executar validação integral + benchmark + governança (decisão adotada)**
   - Prós: maior rastreabilidade, evidência auditável e menor risco de regressão silenciosa.
   - Contras: custo computacional ligeiramente maior.

## Consequências
- O projeto mantém baseline técnico e documental sincronizados no mesmo ciclo.
- O processo de aprovação humana passa a ter evidências objetivas e reproduzíveis.
- Warnings operacionais permanecem visíveis para evolução incremental, sem bloquear entrega quando não há falha funcional.
