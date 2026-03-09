# Export (PyTorch -> ONNX -> TensorFlow)

## Visão arquitetural

Treino e pesquisa permanecem em PyTorch/PyFolds. A interoperabilidade é feita por:

1. `pyfolds.export.export_to_onnx`
2. `pyfolds.export.convert_onnx_to_tf`
3. `pyfolds.export.validate_tf_saved_model`

## Limitações

- Conversão ONNX->TF depende de `onnx-tf` e cobertura de operadores.
- Alguns grafos dinâmicos podem falhar no export.
