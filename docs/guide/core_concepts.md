# Guide — Core Concepts

## Modelo mental

PyFolds decompõe o cálculo neural em camadas funcionais:

- **Sinapse (estado):** `N`, `I`, `W`
- **Dendrito (subunidade):** integração local
- **Soma (decisor):** `u`, `theta`, `spike`
- **Axônio (saída):** emissão e fases de processamento

## Por que isso importa?

Essa decomposição permite depuração causal e reduz opacidade: você observa estados locais em vez de apenas tensores internos agregados.
