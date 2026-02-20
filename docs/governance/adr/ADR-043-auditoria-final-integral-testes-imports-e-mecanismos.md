# ADR-043 — Auditoria final integral de testes, imports e mecanismos

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia de Qualidade e Manutenção de Runtime
- **Contexto:** O projeto requer uma trilha auditável para executar verificações de regressão abrangentes (testes, superfície de importação pública e mecanismos operacionais), com execução repetível e baixo risco de omissão manual.

## Contexto

Mesmo com suíte extensa de testes, auditorias pontuais tendem a variar de acordo com quem executa o processo. Isso gera dois riscos:

1. **Falso verde parcial:** rodar só parte dos testes e concluir estabilidade.
2. **Quebra de contratos públicos:** mudanças no `__all__` ou exports sem validação explícita.

Além disso, há necessidade de um procedimento objetivo para reprodução rápida em incidentes e revisões de release.

## Decisão

Padronizar a auditoria final em um roteiro executável único, `scripts/run_full_audit.sh`, com ordem obrigatória:

1. **Validação de compilação/import sintático** via `compileall`.
2. **Validação da superfície pública de imports** verificando `pyfolds.__all__`.
3. **Smoke test de instalação/execução** via `test_install.py`.
4. **Execução integral da suíte de testes** via `pytest -q`.

## Algoritmo operacional (passo a passo)

```text
INÍCIO
  executar compileall em src/, tests/ e run_pyfolds.py
  se falhar: interromper, abrir incidente com traceback

  carregar pyfolds e iterar pyfolds.__all__
  se algum símbolo não existir: interromper, corrigir export/import

  executar test_install.py
  se falhar: classificar como falha de empacotamento/contrato de instalação

  executar pytest -q (suíte completa)
  se falhar: priorizar correção por camada (core > integração > periféricos)

  publicar evidências em PR (comandos + resultados)
FIM
```

## Consequências

### Positivas
- Processo único e reproduzível para auditoria final.
- Redução de divergências entre execução local e validação de CI.
- Evidência objetiva para aprovação de mudanças sensíveis.

### Negativas / Trade-offs
- Tempo de execução maior em comparação com checks parciais.
- Dependência de ambiente com dependências já resolvidas para rodar tudo em sequência.

## Alternativas consideradas

1. **Manter apenas execução manual ad hoc:** rejeitada por baixa confiabilidade.
2. **Executar apenas `pytest`:** rejeitada por não validar explicitamente contrato de import público e smoke de instalação.

## Referências

- `scripts/run_full_audit.sh`
- `tests/unit/test_public_import_surface.py`
- `test_install.py`
