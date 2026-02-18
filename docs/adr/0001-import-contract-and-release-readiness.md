# ADR 0001 — Contrato de import público e prontidão de release

- **Status:** Aceito
- **Data:** 2026-02-18

## Contexto

Após evolução recente do pacote, precisamos reduzir risco de quebra em aplicações externas que usam:

```python
from pyfolds import ...
```

Também precisamos aumentar confiança de publicação no PyPI cobrindo:
1. Import surface pública.
2. Funcionamento básico de classes nucleares.
3. Evidência de exemplos para treinamento (MNIST/CIFAR) com padrão de import oficial.

## Decisão

1. Criar testes de smoke para `pyfolds.__all__`, garantindo que todo símbolo exportado esteja acessível.
2. Validar instanciação de classes core (`MPJRDConfig`, `MPJRDNeuron`, `MPJRDLayer`, `MPJRDNetwork`) e fluxo básico de telemetria.
3. Documentar a API ativa e o padrão oficial de imports.
4. Incluir script de referência para treinamento MNIST/CIFAR em pasta de testes para análise posterior.

## Consequências

### Positivas
- Menor chance de quebra silenciosa em consumidores externos.
- Pipeline de release ganha evidência objetiva de integridade de imports.
- Equipe tem referência centralizada de API pública ativa.

### Riscos / Limitações
- Teste de smoke não substitui benchmark funcional completo.
- Script de treinamento de datasets pode exigir download externo quando executado.

## Implementação ligada a esta ADR

- `tests/unit/test_public_import_surface.py`
- `docs/guides/imports-and-active-api.md`
- `tests/training/mnist_cifar_training_reference.py`
