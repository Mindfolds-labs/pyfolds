# FOLDSNet Architecture Report

Este relatório resume topologia por camada e custo aproximado do classificador final (FP32).

## Fluxo de camadas

```mermaid
flowchart LR
    A[Input] --> B[Retina]
    B --> C[LGN]
    C --> D[V1]
    D --> E[IT]
    E --> F[Linear Classifier]
```

## Tabela comparativa

| dataset | variant | input_shape | classes | retina | lgn | v1 | it | total_neurons | classifier_params | classifier_mem_mib_fp32 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnist | 2L | (1, 28, 28) | 10 | 10 | 10 | 20 | 10 | 50 | 110 | 0.00042 |
| mnist | 4L | (1, 28, 28) | 10 | 49 | 49 | 98 | 49 | 245 | 500 | 0.001907 |
| mnist | 5L | (1, 28, 28) | 10 | 64 | 64 | 128 | 64 | 320 | 650 | 0.00248 |
| mnist | 6L | (1, 28, 28) | 10 | 84 | 84 | 168 | 84 | 420 | 850 | 0.003242 |
| cifar10 | 2L | (3, 32, 32) | 10 | 39 | 39 | 78 | 39 | 195 | 400 | 0.001526 |
| cifar10 | 4L | (3, 32, 32) | 10 | 192 | 192 | 384 | 192 | 960 | 1930 | 0.007362 |
| cifar10 | 5L | (3, 32, 32) | 10 | 250 | 250 | 500 | 250 | 1250 | 2510 | 0.009575 |
| cifar10 | 6L | (3, 32, 32) | 10 | 327 | 327 | 654 | 327 | 1635 | 3280 | 0.012512 |
| cifar100 | 2L | (3, 32, 32) | 100 | 39 | 39 | 78 | 39 | 195 | 4000 | 0.015259 |
| cifar100 | 4L | (3, 32, 32) | 100 | 192 | 192 | 384 | 192 | 960 | 19300 | 0.073624 |
| cifar100 | 5L | (3, 32, 32) | 100 | 250 | 250 | 500 | 250 | 1250 | 25100 | 0.095749 |
| cifar100 | 6L | (3, 32, 32) | 100 | 327 | 327 | 654 | 327 | 1635 | 32800 | 0.125122 |

## Notas

- O número de neurônios em Retina/LGN/V1/IT escala com a variante (`2L < 4L < 5L < 6L`).
- O custo de parâmetros explícitos aqui cobre principalmente o `nn.Linear` final.
- A dinâmica bioinspirada principal está nos neurônios/camadas MPJRD usados dentro do FOLDSNet.
- Mecanismos de EEG, engram e memória de consolidação são mais diretamente configuráveis no caminho MPJRD avançado, não como bloco dedicado no classificador FOLDSNet.
