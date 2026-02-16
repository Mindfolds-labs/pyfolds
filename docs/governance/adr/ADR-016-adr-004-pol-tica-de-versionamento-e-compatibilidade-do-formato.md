# ADR-004 — Política de versionamento e compatibilidade do formato

## Status
Accepted

## Contexto
A evolução contínua do formato exige critérios explícitos para mudanças compatíveis e incompatíveis, evitando quebra silenciosa de leitores antigos.

## Decisão
Definir política de compatibilidade:

- Mudança incompatível estrutural exige novo `magic` ou migração formal documentada.
- Mudança aditiva em índice/chunks deve preservar leitura de versões anteriores.
- Validações de segurança não podem ser relaxadas por compatibilidade retroativa.
- Todo ajuste de formato deve registrar ADR e atualização de spec.

## Consequências
### Positivas
- Evolução previsível e auditável.
- Menor risco de fragmentação de implementações.

### Trade-offs
- Processo de mudança mais disciplinado e com overhead documental.
