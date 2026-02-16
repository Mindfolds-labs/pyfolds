# Benchmarking de serialização

Este guia descreve como medir desempenho de serialização do PyFolds (`.fold/.mind`) com foco em:

- **Throughput de escrita** (MiB/s)
- **Throughput de leitura** (MiB/s)
- **Taxa de compressão** (preferencialmente `fold:zstd` vs `none`, com fallback `zlib(level=6)` quando `zstandard` não estiver disponível)

## Execução manual

A partir da raiz do repositório:

```bash
python scripts/run_benchmarks.py --output docs/assets/benchmarks_results.json
python scripts/generate_benchmarks_doc.py --input docs/assets/benchmarks_results.json --output docs/BENCHMARKS.md
```

Parâmetros úteis:

- `--samples`: número de amostras por medição (mediana). Exemplo:

```bash
python scripts/run_benchmarks.py --samples 7 --output docs/assets/benchmarks_results.json
```

## Saídas

- JSON determinístico: `docs/assets/benchmarks_results.json`
- Relatório em Markdown: `docs/BENCHMARKS.md`

## Como interpretar os números

1. **Throughput (MiB/s)**
   - Maior = melhor desempenho.
   - Compare `none` e `zstd` separadamente para escrita/leitura.

2. **Razão de compressão (`ratio_vs_none`)**
   - `1.0` = mesmo tamanho do baseline `none`.
   - `< 1.0` = compressão efetiva (menor uso de disco).
   - `> 1.0` = pior que baseline.
   - Consulte também o campo `method` no JSON para saber se a medição veio de `fold:zstd` ou de fallback `zlib(level=6)`.

3. **Redução de espaço (%)**
   - Positivo = economia de espaço.
   - Negativo = aumento de tamanho.

## Automação no GitHub Actions

O workflow `.github/workflows/benchmarks.yml` executa semanalmente (segunda-feira, 06:00 UTC), regenera os arquivos e faz commit automático quando houver mudança nos resultados.
