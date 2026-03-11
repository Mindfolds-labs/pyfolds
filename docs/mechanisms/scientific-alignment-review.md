# Scientific Alignment Review

## Objetivo
Documentar o nível de aderência entre hipóteses neurocientíficas e implementações computacionais atuais.

## Variáveis
- **Entrada:** mecanismos ativos, evidências observáveis e literatura de referência.
- **Controle:** escopo de revisão por família de mecanismo.
- **Saída:** classificação de aderência e riscos de interpretação.

## Fluxo
1. Mapear mecanismo para implementação concreta.
2. Identificar simplificações/modelagens adotadas.
3. Registrar lacunas para validação futura.

## Custo computacional
Custo analítico/documental; quando automatizado usa somente coleta de métricas já existente.

## Integração
- `compare_mechanism_vs_baseline` e `diff_output_stats` (`src/pyfolds/advanced/experimental.py`).
- `NoeticCore` para mecanismos de memória/engram (`src/pyfolds/advanced/noetic_model.py`).
- `MPJRDNeuron` snapshots de fase/poda/conectividade (`src/pyfolds/core/neuron.py`).

## Estado
- **Rótulo:** `Estável`.
- **Justificativa:** é um artefato de governança técnica apoiado por interfaces já disponíveis.
