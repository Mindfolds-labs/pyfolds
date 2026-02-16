# Benchmarks

Relatório gerado automaticamente por `scripts/run_benchmarks.py`.

## Resumo

- Data (UTC): `2026-02-16T13:22:24.058011+00:00`
- Iterações: `5`
- Python: `3.10.19`
- PyTorch: `2.10.0+cu128`
- Plataforma: `Linux-6.12.47-x86_64-with-glibc2.39`

## Velocidade de escrita

| Métrica | Valor |
|---|---:|
| Tempo médio | 1.090717 s |
| Tempo mínimo | 0.980758 s |
| Tempo máximo | 1.340406 s |
| Ops/s | 0.92 |
| Throughput | 3.91 MB/s |

## Velocidade de leitura

| Métrica | Valor |
|---|---:|
| Tempo médio | 3.395471 s |
| Tempo mínimo | 3.345210 s |
| Tempo máximo | 3.452377 s |
| Ops/s | 0.29 |
| Throughput | 1.26 MB/s |

## Compressão

| Métrica | Valor |
|---|---:|
| Tamanho sem compressão | 4474818 bytes |
| Tamanho comprimido | 498132 bytes |
| Razão de compressão (raw/compressed) | 8.983x |
| Economia de espaço | 88.87% |
| Método | zlib-fallback |

## Fonte

- JSON completo: `docs/assets/benchmarks_results.json`
