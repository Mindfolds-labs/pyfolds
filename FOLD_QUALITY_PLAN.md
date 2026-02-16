# FOLD/.MIND Quality Plan

## Objetivo
Transformar o formato de serialização `.fold/.mind` em um artefato publicável e auditável com:
- especificação normativa;
- ADRs para decisões arquiteturais;
- suíte de robustez para entradas corrompidas;
- benchmarks reprodutíveis com publicação contínua.

## Fora de escopo
- refatoração ampla de módulos não relacionados à serialização;
- mudanças de API pública fora do fluxo de `.fold/.mind`;
- tuning global de performance sem evidência em benchmark.

## Definição de pronto (DoD)
Uma entrega é considerada pronta quando todos os itens abaixo forem verdadeiros:
- documentação adicionada com escopo e linguagem normativa consistentes;
- testes novos passam localmente sem dependências adicionais;
- workflow de benchmark executa por demanda e agenda semanal;
- evidências anexadas no PR (checklist preenchido + logs/artefatos);
- mudanças confinadas à superfície permitida da ação.

## Ordem de merge recomendada
1. `chore/fold-quality-tracking`
2. `docs/fold-spec-and-adrs`
3. `test/fold-corruption-suite`
4. `bench/continuous-benchmarks`

## Dependências e sequência
- Ação 2 depende da conclusão de escopo/DoD da Ação 1.
- Ação 3 depende da existência de requisitos de validação (Ação 2).
- Ação 4 depende de casos de uso estabilizados e, preferencialmente, testes de robustez ativos (Ação 3).

## Critérios de aceite por ação

### Ação 1 — Registro do trabalho
- [ ] `FOLD_QUALITY_PLAN.md` criado na raiz.
- [ ] `RISK_REGISTER.md` criado na raiz.
- [ ] Nenhum arquivo fora da raiz alterado.

### Ação 2 — Especificação e ADRs
- [ ] `docs/spec/FOLD_SPECIFICATION.md` criado.
- [ ] ADR template + ADRs iniciais adicionados em `docs/adr/`.
- [ ] Linguagem normativa com MUST/SHOULD/MAY.
- [ ] Nenhuma alteração fora de `docs/spec/` e `docs/adr/`.

### Ação 3 — Testes de robustez
- [ ] Novos testes de corrupção/truncamento/limites adicionados em `tests/`.
- [ ] Falha rápida em cenários adversariais (ValueError/EOFError/RuntimeError).
- [ ] Execução determinística sem flakiness óbvia.
- [ ] Nenhuma alteração fora de `tests/`.

### Ação 4 — Benchmarks e publicação
- [ ] Benchmarks em `benchmarks/bench_foldio.py`.
- [ ] Workflow em `.github/workflows/benchmarks.yml`.
- [ ] Artifact `benchmark.json` gerado.
- [ ] Gatilhos limitados (`workflow_dispatch` e `schedule`).
- [ ] Nenhuma alteração fora de `benchmarks/` e `.github/workflows/`.

## Comandos de validação por ação (somente execução)

### Ação 1
```bash
git status --short
```

### Ação 2
```bash
pytest -q tests/unit/serialization/test_foldio.py
```

### Ação 3
```bash
pytest -q tests/unit/serialization
```

### Ação 4
```bash
pytest benchmarks/bench_foldio.py --benchmark-only --benchmark-json=benchmark.json
```

## Checklist de evidências por PR
- [ ] Escopo da ação confirmado.
- [ ] Riscos relevantes avaliados.
- [ ] Comandos de validação executados.
- [ ] Logs/artefatos anexados.
- [ ] Estratégia de rollback descrita.
