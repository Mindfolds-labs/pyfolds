# Benchmarks de Serialização (FoldIO)

- Gerado em: `2026-02-16T13:33:21+00:00`
- Seed: `1337`
- Amostras por medição: `3`
- Device: `cpu`
- Python: `3.10.19`
- PyTorch: `2.10.0+cu128`
- Compressão Fold/ZSTD disponível: `False`

## Throughput de escrita

| Cenário | none (MiB/s) | Arquivo none (bytes) |
|---|---:|---:|
| small (4x16, batch=16) | 1.454 | 149357 |
| medium (8x32, batch=32) | 2.291 | 565173 |

## Throughput de leitura

| Cenário | none (MiB/s) |
|---|---:|
| small (4x16, batch=16) | 0.778 |
| medium (8x32, batch=32) | 0.732 |

## Taxa de compressão

| Cenário | Método | Razão vs none | Redução de espaço (%) |
|---|---|---:|---:|
| small | zlib(level=6) | 0.147 | 85.269 |
| medium | zlib(level=6) | 0.123 | 87.703 |

Interpretação rápida: throughput maior é melhor; razão de compressão menor que 1.0 indica arquivo comprimido menor que o baseline `none`.
