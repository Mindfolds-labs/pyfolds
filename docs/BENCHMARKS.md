# Benchmarks

Resultados automáticos gerados por `scripts/run_benchmarks.py`.

- Tamanho do payload: `16 MB`
- Iterações por codec: `5`
- Codecs avaliados: `none`

| Codec | Write speed (MB/s) | Read speed (MB/s) | Compress ratio | Payload size (bytes) | File size (bytes) |
|---|---:|---:|---:|---:|---:|
| none | 156.1964 | 245.8661 | 0.999963 | 16777216 | 16777841 |

## Reprodução local

```bash
python scripts/run_benchmarks.py
```

Este comando atualiza:
- `docs/assets/benchmarks_results.json`
- `docs/BENCHMARKS.md`
