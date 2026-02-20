---
id: "ISSUE-014"
titulo: "Sanity check de integridade de pesos e atualização documental v2.0.2"
prioridade: "Alta"
area: "Segurança/Runtime/Docs"
fase: "ativa"
adr_vinculada: "ADR-046"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---

# ISSUE-014: Sanity check de integridade de pesos e atualização documental v2.0.2

## Metadados
- **Fase:** `ativa`
- **ADR vinculada:** `ADR-046`

## Objetivo
Implementar uma camada adicional de verificação periódica da integridade de pesos e buffers em runtime e atualizar a documentação de governança da versão `2.0.2`.

## Contexto Técnico
A versão `2.0.2` já introduziu hardening em serialização/checkpoint (ADR-045). Como backlog de especialista, faltava um mecanismo operacional para detectar corrupção silenciosa em execução longa.

## Análise Técnica
1. Criar monitor com hash determinístico SHA-256 do `state_dict`.
2. Permitir baseline explícito (`set_baseline`) e lazy (`check_integrity` inicial).
3. Disponibilizar verificação periódica por `check_every_n_steps`.
4. Expor API pública no pacote.
5. Cobrir cenários positivos/negativos por testes unitários.
6. Vincular decisão arquitetural via ADR e registrar a ISSUE no HUB.

## Requisitos Funcionais
- [x] RF-01 Implementar `ModelIntegrityMonitor`.
- [x] RF-02 Expor classe em `pyfolds.monitoring` e `pyfolds`.
- [x] RF-03 Detectar drift entre hash esperado e hash atual.

## Requisitos Não-Funcionais
- [x] RNF-01 Performance: checagem por intervalo configurável.
- [x] RNF-02 Segurança: hash criptográfico para detectar alteração inesperada.

## Artefatos Esperados
- Código
- Testes
- Documentação

## Critérios de Aceite
- [x] Código funcional implementado.
- [x] Testes cobrindo baseline e mutação inesperada.
- [x] Fase validada contra `docs/development/WORKFLOW_INTEGRADO.md`.
- [x] Referência ADR registrada quando houver mudança estrutural.

## Riscos e Mitigações
- **Risco:** falsos positivos durante treino ativo.
- **Mitigação:** usar baseline em janelas estáveis ou após checkpoints de referência.

## PROMPT:EXECUTAR
```yaml
issue_id: "ISSUE-014"
objetivo: "Implementar sanity check periódico de integridade de pesos e atualizar docs v2.0.2"
fase: "ativa"
adr_vinculada: "ADR-046"
```

## Rastreabilidade (IEEE 830)
| Requisito | Evidência |
| --- | --- |
| RF-01 | `src/pyfolds/monitoring/health.py` |
| RF-02 | `src/pyfolds/monitoring/__init__.py`, `src/pyfolds/__init__.py` |
| RF-03 | `tests/unit/core/test_monitoring_and_checkpoint.py` |
