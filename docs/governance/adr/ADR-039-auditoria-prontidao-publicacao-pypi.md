# ADR-039 — Auditoria de prontidão para publicação no PyPI (ISSUE-038)

## Status
Aceito

## Contexto
Foi solicitada uma auditoria sênior para verificar se o pacote `pyfolds` está pronto para publicação no PyPI com base em checklist de engenharia: estrutura do projeto, metadados de empacotamento, validação de build, qualidade, testes, segurança e conformidade de distribuição.

A execução técnica confirmou sucesso em `python -m build`, `twine check dist/*` e `PYTHONPATH=src pytest -q`, mas revelou alertas de governança de metadados (`setuptools`) para evolução futura.

## Decisão
1. Considerar o pacote **apto para publicação técnica imediata** no PyPI (sem bloqueios críticos de build/teste).
2. Manter como gate mínimo de release os comandos:
   - `python -m build`
   - `twine check dist/*`
   - `PYTHONPATH=src pytest -q`
3. Registrar como dívida técnica prioritária para próximo ciclo:
   - migrar `project.license` para expressão SPDX em string;
   - mover `classifiers` e `keywords` para `pyproject.toml`;
   - reduzir duplicidade entre `pyproject.toml` e `setup.cfg`.

## Alternativas consideradas
1. **Bloquear publicação até eliminar todos os warnings de `setuptools`**
   - Prós: aderência máxima imediata às recomendações futuras.
   - Contras: atraso desnecessário de release sem falha funcional atual.

2. **Permitir publicação e tratar warnings em hardening dedicado (decisão adotada)**
   - Prós: mantém cadência de entrega com risco controlado.
   - Contras: mantém dívida técnica temporária de metadados.

## Consequências
- O projeto ganha trilha formal de auditoria PyPI e critério objetivo de “go/no-go”.
- Publicações futuras ficam mais previsíveis com gate mínimo explícito.
- É criado backlog técnico claro para adequação completa ao padrão moderno de empacotamento.
