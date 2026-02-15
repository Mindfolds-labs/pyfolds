# Contributing to PyFolds

Obrigado por contribuir com o PyFolds! Este guia resume o fluxo recomendado para contribuições.

## 1. Setup de desenvolvimento

```bash
git clone https://github.com/Mindfolds-labs/pyfolds.git
cd pyfolds
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 2. Padrões de código

- **PEP 8** para estilo Python.
- Use **type hints** em funções públicas e novos módulos.
- Escreva docstrings curtas e objetivas para classes/métodos.
- Prefira nomes explícitos para variáveis relacionadas a dinâmica neural (`spike_rate`, `theta`, `saturation_ratio`).

## 3. Testes

Rode os testes antes de abrir PR:

```bash
pytest
```

Opcional (subconjuntos):

```bash
pytest tests/unit
pytest tests/integration
pytest tests/performance
```

## 4. Fluxo de contribuição

1. Crie uma branch a partir de `main`.
2. Faça commits pequenos e descritivos.
3. Atualize documentação quando alterar comportamento público.
4. Abra PR com:
   - contexto do problema;
   - solução aplicada;
   - impacto esperado;
   - evidências (testes/benchmarks).

## 5. Checklist de PR

- [ ] Código formatado e legível.
- [ ] Testes passando localmente.
- [ ] Sem quebras de API não documentadas.
- [ ] README/docs atualizados, se aplicável.
- [ ] Changelog atualizado para mudanças relevantes.
