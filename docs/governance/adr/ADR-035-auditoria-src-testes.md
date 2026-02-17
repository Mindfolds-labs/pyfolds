# ADR-035 — Auditoria de `src/`, testes e estratégia de correção incremental

## Status
Proposto

## Contexto
A ISSUE-012 executou auditoria focada em `src/` com evidências reprodutíveis em logs para: compilação, importação, instalação editável, validações de documentação/hub e suíte de testes.

Principais achados:
- `python -m compileall src/` sem erros de sintaxe.
- Smoke import de `pyfolds` funcionando com `PYTHONPATH=src`.
- `pip install -e .` falha no modo padrão (build isolation) por restrição de proxy/índice para dependência de build (`setuptools>=61.0`).
- `pip install -e . --no-build-isolation` conclui com sucesso no ambiente auditado.
- Testes passam (`198 passed`), com warnings de qualidade (marker não registrado e warning controlado de cleanup em teste de serialização).

## Decisão
Adotar estratégia de correção **incremental por issues pequenas**, priorizadas por severidade:
- **P0**: problemas que bloqueiam instalação/execução em cenários reais (ex.: fluxo de instalação sob rede restrita).
- **P1**: falhas funcionais de teste/comportamento (não observadas nesta auditoria).
- **P2**: melhorias de qualidade/manutenção com evidência (warnings e ruídos operacionais).

As correções devem ser separadas em PRs curtos, cada um com reprodução mínima e validação objetiva.

## Alternativas consideradas
1. **Corrigir tudo em um único PR grande**
   - Prós: fechamento rápido em volume.
   - Contras: alto risco de regressão, revisão difícil, baixa rastreabilidade causa→efeito.

2. **Correções incrementais por prioridade (decisão adotada)**
   - Prós: menor risco, rollback simples, observabilidade por issue, governança clara.
   - Contras: demanda disciplina de planejamento e acompanhamento contínuo.

## Consequências
- Melhor previsibilidade de entrega e revisão.
- Maior capacidade de associar cada correção ao achado auditado.
- Necessidade de manter fila de execução atualizada com status e vínculos de evidência.
