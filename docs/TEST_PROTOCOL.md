# TEST_PROTOCOL — Validação científica do MPJRD

## 1) Objetivo

Avaliar desempenho, robustez e estabilidade interna do modelo dendrítico sem camadas ocultas explícitas.

## 2) Benchmarks

- **MNIST padrão** (classificação de dígitos).
- **MNIST + ruído Gaussiano** em múltiplos níveis de \(\sigma\) para robustez.
- (Opcional) Fashion-MNIST para generalização cruzada.

## 3) Comparadores

1. Perceptron linear (mesmo orçamento aproximado de parâmetros).
2. MLP raso (1 camada oculta) como referência superior.
3. MPJRD completo.
4. MPJRD com ablação (sem WTA ou sem não linearidade local).

## 4) Métricas

- Externas: acurácia, macro-F1, AUROC (quando aplicável).
- Robustez: curva acurácia vs. \(\sigma\) do ruído.
- Internas: `spike_rate`, `theta`, `r_hat`, `saturation_ratio`, `protection_ratio`.

## 5) Procedimento

1. Fixar seeds (>= 3 seeds).
2. Treinar todos os modelos com mesmo protocolo de epochs.
3. Avaliar em clean + ruído.
4. Rodar ablações estruturais.
5. Consolidar resultados com média ± desvio padrão.

## 6) Critérios de aprovação

- MPJRD > baseline linear em cenário clean.
- Menor degradação relativa sob ruído moderado.
- Estabilidade homeostática sem colapso (neurônio morto ou saturação total).

## 7) Checklist de execução

- Registrar commit hash.
- Salvar configuração completa de experimento.
- Persistir logs de telemetria por etapa.
- Exportar tabela final para comparação estatística.
