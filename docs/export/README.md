# Export (PyTorch -> ONNX -> TensorFlow)

## Visão arquitetural

Treino e pesquisa permanecem em PyTorch/PyFolds. A interoperabilidade é feita por:

1. `pyfolds.export.export_to_onnx`
2. `pyfolds.export.convert_onnx_to_tf`
3. `pyfolds.export.validate_tf_saved_model`
4. `pyfolds.mobile.convert_saved_model_to_tflite` / `pyfolds.mobile.validate_tflite_model`

As rotinas de export mobile retornam um relatório estruturado com diagnóstico do `.tflite`
(inputs/outputs, tipos, operadores, quantização, discrepância numérica quando há SavedModel de referência)
e classificação de confiabilidade.

## Classificação de confiabilidade (TFLite)

A classificação retornada no relatório segue critérios objetivos:

- **`confiável`**: sem ops custom/Flex, I/O final em `float32`, sem flags de degradação.
- **`experimental`**: presença de ops custom/Flex ou dtype final não `float32`.
- **`degradado semanticamente`**: quantização agressiva (`int8`/`uint8`) em I/O ou discrepância
  numérica máxima acima do limiar de validação.

## Matriz de cobertura A/B/C por componente

| Componente | Nível | Escopo atual | Risco de degradação semântica |
| --- | --- | --- | --- |
| `export_to_onnx` | **A** | Export de modelos compatíveis com ONNX para uso de inferência. | Baixo a médio: diferenças de shape dinâmico e normalização podem alterar numericamente saídas limite. |
| `convert_onnx_to_tf` | **B** | Conversão ONNX -> TensorFlow para fluxos suportados por `onnx-tf`. | Médio a alto: cobertura parcial de operadores pode exigir fallback, simplificação de grafo ou ajuste manual. |
| `validate_tf_saved_model` | **B** | Validação básica estrutural/funcional do SavedModel exportado. | Médio: validação não implica equivalência total de comportamento em todos os domínios de entrada. |
| Recursos de plasticidade avançada no grafo convertido | **C (não suportado)** | Não há promessa de paridade para estados/atualizações plásticas além da inferência padrão. | Alto: semânticas avançadas podem ser perdidas, aproximadas ou removidas no pipeline de export. |

### Legenda de níveis

- **A**: suportado operacionalmente no fluxo principal de inferência.
- **B**: suportado com ressalvas, sujeito a lacunas de operadores e ajustes por caso.
- **C**: não suportado; sem garantia de equivalência funcional/semântica.

## Limitações

- Conversão ONNX->TF depende de `onnx-tf` e cobertura de operadores.
- Alguns grafos dinâmicos podem falhar no export.
- A validação atual verifica viabilidade de execução, não equivalência matemática plena entre runtimes.

## Não suportado (nível C)

Para evitar promessa indevida de equivalência entre stacks, este fluxo **não** garante:

- equivalência semântica completa PyTorch ↔ ONNX ↔ TensorFlow;
- manutenção automática de comportamentos de plasticidade avançada;
- cobertura integral de operadores não padronizados/custom.

## Pendências técnicas

- Lacuna registrada: ausência de ADR específico para decisões de interoperabilidade TensorFlow (escopo, critérios de equivalência e limites de suporte).
