# ADR-0002 — Container `.fold/.mind` para checkpoints científicos

## Status
Accepted

## Contexto
`torch.save()` isolado atende retomada de treino, mas não resolve bem:
- inspeção parcial sem desserializar tensores;
- integridade por partes (corrupção localizada);
- telemetria e auditoria no mesmo artefato;
- metadados explícitos para reprodutibilidade.

## Decisão
Padronizar um **container único** com duas extensões:
- `.fold`: padrão técnico (infra/storage);
- `.mind`: branding para artefatos com chunks de IA (`ai_graph`/`ai_vectors`).

Formato físico idêntico, versão atual `1.2.0`.

## Layout
1. Header fixo (`magic`, offset e tamanho do índice)
2. Chunks tipados (`torch_state`, `nuclear_arrays`, `llm_manifest`, `metrics`, `history`, `telemetry`)
3. Índice JSON com offsets e hashes por chunk

Integridade:
- CRC32C (rápido) + SHA-256 (robusto) por chunk
- ECC opcional por chunk (`none` ou Reed-Solomon `rs(n)`)

## Segurança
- validação de CRC/SHA antes de descompressão e desserialização;
- carregamento torch em modo seguro por padrão (`weights_only=True`);
- modo "trusted" explícito para ambientes controlados.

## Reprodutibilidade
Manifesto e metadados incluem:
- versão do formato;
- hash git (best effort);
- versão Python/Torch/plataforma;
- `torch_initial_seed`.

## Consequências
### Positivas
- leitura parcial via índice + mmap;
- melhor diagnóstico em corrupção localizada;
- auditoria integrada com `llm_manifest` + telemetria.

### Trade-offs
- maior complexidade em relação a `torch.save` simples;
- ECC aumenta custo de CPU/disco quando habilitado.

## Evolução planejada
- chunk `nuclear_arrays` para análise científica sem dependência de `torch_state` (já incorporado no padrão v1.2);
- chunk `ai_graph`/`ai_vectors` para declarar arquivo como `.mind`.
