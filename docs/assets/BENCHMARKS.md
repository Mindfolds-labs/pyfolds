# Benchmarks de Serialização (FoldIO)

- Gerado em: `2026-02-17T17:14:53+00:00`
- Seed: `1337`
- Amostras por medição: `5`
- Device: `cpu`
- Python: `3.12.12`
- PyTorch: `2.10.0+cu128`
- Compressão Fold/ZSTD disponível: `False`

## Throughput de escrita

| Cenário | none (MiB/s) | Arquivo none (bytes) |
|---|---:|---:|
| small (4x16, batch=16) | 2.197 | 149357 |
| medium (8x32, batch=32) | 2.821 | 565173 |

## Throughput de leitura

| Cenário | none (MiB/s) |
|---|---:|
| small (4x16, batch=16) | 0.926 |
| medium (8x32, batch=32) | 0.959 |

## Taxa de compressão

| Cenário | Método | Razão vs none | Redução de espaço (%) |
|---|---|---:|---:|
| small | zlib(level=6) | 0.147 | 85.268 |
| medium | zlib(level=6) | 0.123 | 87.703 |

Interpretação rápida: throughput maior é melhor; razão de compressão menor que 1.0 indica arquivo comprimido menor que o baseline `none`.
