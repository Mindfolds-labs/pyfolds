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

## Limitações

- Conversão ONNX->TF depende de `onnx-tf` e cobertura de operadores.
- Alguns grafos dinâmicos podem falhar no export.
