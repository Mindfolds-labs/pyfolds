# ADR-036 — Governança de validação integral pós-correção (import + testes + issue)

## Status
Aceito

## Contexto
Após a correção de erros semânticos em módulos avançados, houve insatisfação com a rastreabilidade e com a garantia operacional de que o pacote continua importável/executável sem ajustes manuais de contexto.

Também houve solicitação explícita para formalizar a execução com:
- issue operacional com artefatos e riscos;
- execução de verificações completas possíveis no ambiente;
- confirmação explícita da importação de `pyfolds`.

## Decisão
Padronizar, para correções de estabilidade/runtime, o fluxo mínimo de validação operacional:
1. `python -m compileall src`;
2. `python -m pip install -e . --no-build-isolation` (quando necessário no ambiente);
3. smoke import (`import pyfolds` e símbolos core);
4. execução da suíte padrão (`pytest -q`);
5. criação/atualização de ISSUE em `docs/development/prompts/relatorios/`;
6. sincronização e checagem do hub (`tools/sync_hub.py` + `--check`);
7. validação de formato/link das issues.

## Alternativas consideradas
1. **Validar apenas módulos alterados**
   - Prós: mais rápido.
   - Contras: risco de regressão transversal não detectada.

2. **Validar suíte completa + governança documental (decisão adotada)**
   - Prós: maior segurança operacional e rastreabilidade.
   - Contras: custo maior de execução.

## Consequências
- Aumenta confiança em correções críticas de runtime.
- Reduz ambiguidades de ambiente para importação (`pyfolds`).
- Exige disciplina de atualização da fila e validações de governança.
