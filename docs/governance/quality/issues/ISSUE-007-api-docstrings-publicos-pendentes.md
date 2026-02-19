# ISSUE-007 — Símbolos públicos sem docstring no contrato de neurônio

## Contexto
Durante a auditoria, foi executado:

```bash
python tools/check_api_docs.py
```

A verificação finalizou, porém reportou símbolos públicos sem docstring em `src/pyfolds/contracts/neuron_contract.py`.

## Evidência
Pendências identificadas:

- `MechanismStep`
- `NeuronStepInput`
- `StepExecutionTrace`
- `NeuronStepOutput`
- `ContractViolation`
- `validate_step_output`

## Impacto
- Documentação de API pública incompleta.
- Aumento da curva de adoção e risco de uso incorreto dos contratos.
- Menor qualidade de geração de docs automatizadas.

## Critérios de aceite
- Adicionar docstrings nos símbolos reportados.
- Manter consistência com padrão de documentação do projeto.
- `python tools/check_api_docs.py` sem pendências.
