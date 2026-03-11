# TensorFlow Serving

## Escopo

Este guia cobre **deploy e inferência** com TensorFlow Serving para modelos previamente exportados:

1. Prepare layout versionado com `prepare_versioned_saved_model`.
2. Gere comando local com `build_tf_serving_command`.
3. Monte payload JSON com `build_inference_payload`.

O foco é operação de serving (empacotamento, versionamento e chamada de inferência), não equivalência total de semântica entre frameworks.

## Garantias e limites

- Suporte voltado ao caminho operacional de deploy/inferência em TF Serving.
- Não há garantia de equivalência matemática completa entre PyTorch, ONNX e TF Serving para todos os casos.
- Não há garantias de suporte a plasticidade avançada no ambiente de serving.

## Não suportado (nível C)

Para evitar promessa indevida de equivalência, é classificado como **nível C (não suportado)**:

- reprodução integral de comportamentos de treino/pesquisa no runtime de serving;
- semântica completa de mecanismos avançados de plasticidade;
- casos fora do pipeline de exportação e inferência explicitamente suportado.

## Pendências técnicas

- Lacuna registrada: ausência de ADR TF consolidando limites de suporte, critérios de validação e definição formal de equivalência.
