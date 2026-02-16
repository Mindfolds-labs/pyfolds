# ADR-001 — Formalização da especificação `.fold/.mind`

## Status
Accepted

## Contexto
O formato `.fold/.mind` já era usado em produção, mas a documentação estava dispersa entre notas teóricas e detalhes de implementação. Isso dificultava auditoria, onboarding e evolução controlada do parser.

## Decisão
Adotar uma especificação normativa única em `docs/FOLD_SPECIFICATION.md` contendo:

- offsets binários exatos;
- endianness formal;
- layout header/index/chunks;
- regras de validação anti-corrupção;
- limites anti-DoS (`MAX_INDEX_SIZE` e `MAX_CHUNK_SIZE`);
- algoritmo de leitura passo a passo.

## Consequências
### Positivas
- Melhora rastreabilidade entre implementação e documentação.
- Facilita revisão de segurança e interoperabilidade futura.
- Reduz ambiguidade no comportamento esperado do parser.

### Trade-offs
- Requer manutenção sincronizada entre código e spec.
