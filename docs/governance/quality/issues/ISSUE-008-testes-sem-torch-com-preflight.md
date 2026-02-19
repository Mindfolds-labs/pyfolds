# ISSUE-008 — Suite de testes não degrada graciosamente sem PyTorch

## Contexto
Durante execução da auditoria funcional em ambiente mínimo:

```bash
pytest -q
```

A execução falhou já no carregamento de `tests/conftest.py` devido a import obrigatório de `torch`.

## Evidência
Erro observado:

```text
ImportError while loading conftest '/workspace/pyfolds/tests/conftest.py'.
tests/conftest.py:12: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
```

## Impacto
- Impossibilidade de rodar qualquer subconjunto de testes sem stack completa instalada.
- Auditorias automáticas em ambientes enxutos falham cedo, sem separar “falha de código” de “falha de ambiente”.
- Menor previsibilidade para contribuição externa.

## Proposta
- Adicionar preflight explícito para dependências em testes (mensagem clara e instrução de instalação).
- Onde aplicável, aplicar `pytest.importorskip("torch")` em módulos que dependem de PyTorch.
- Definir alvo de teste “mínimo sem torch” para validações não-ML (ferramentas/docs).

## Critérios de aceite
- Falhas por ausência de dependência são reportadas de forma clara e intencional (skip/erro guiado).
- Comando de testes de docs/tools roda sem exigir `torch`.
- Documentação de contribuição explicita matriz de dependências por tipo de teste.
