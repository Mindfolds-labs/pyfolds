# Mobile / Edge

## Escopo

Este módulo cobre **inferência e deploy em dispositivos** a partir de artefatos já exportados:

- `pyfolds.mobile.tflite_converter`: SavedModel -> `.tflite`.
- `noetic_pawp.mobile.android_app`: gera guia mínimo de integração Android.

O objetivo é empacotamento e execução de inferência no edge, não reimplementação completa do comportamento de treino/pesquisa.

## Garantias e limites

- O escopo de suporte é focado em pipeline de deploy mobile e inferência online/offline.
- Não há garantias de equivalência total com o runtime original de treino.
- Não há garantias de suporte a mecanismos de plasticidade avançada além da inferência padrão.

## Não suportado (nível C)

Para evitar promessa indevida de equivalência, considera-se **nível C (não suportado)**:

- paridade funcional completa com funcionalidades avançadas do ambiente de pesquisa;
- atualização dinâmica de estados plásticos complexos em execução mobile;
- qualquer comportamento fora do fluxo de inferência/deploy explicitamente documentado.

## Pendências técnicas

- Lacuna aberta: consolidar ADR de TensorFlow/export/deploy para formalizar critérios de equivalência e fronteiras de suporte.
