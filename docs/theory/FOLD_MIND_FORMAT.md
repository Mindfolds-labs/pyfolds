# Formato científico `.fold/.mind`

## Objetivo
Entregar um artefato único com:
- **retomada de treino** (compatível com PyTorch),
- **análise científica** (metadados e leitura parcial),
- **telemetria integrada**,
- **integridade e recuperação opcional**.

## Especificação resumida
- Magic: `FOLDv1\0\0`
- Header: `magic + header_len + index_off + index_len`
- Chunk header: `type(4) + flags + uncomp_len + comp_len + crc32c + ecc_len`
- Índice JSON no final do arquivo com tabela de chunks e SHA-256 por chunk

## Chunks padrão v1.2
- `torch_state` (`TSAV`): `state_dict`, config, modo, step
- `nuclear_arrays` (`NPZ0`, opcional): arrays científicos (`N`, `I`, `W`, `protection`, `theta`, `r_hat`)
- `llm_manifest` (`JSON`): descrição textual para auditoria e tooling
- `metrics` (`JSON`): saída de `get_metrics()`
- `history` (`JSON`, opcional): histórico do accumulator
- `telemetry` (`JSON`, opcional): snapshot de eventos e stats

## Leitura parcial e escalabilidade
A leitura usa offsets do índice e pode operar com `mmap`, permitindo:
- `peek` de manifesto sem carregar tensores;
- leitura isolada de `nuclear_arrays` para pipelines científicos via `read_nuclear_arrays`.

## Integridade, segurança e correção
- **Detecção**: CRC32C + SHA-256 por chunk;
- **Correção opcional**: ECC Reed-Solomon por chunk;
- **Perfil de proteção**: `off | low | med | high` mapeando para `NoECC`, `rs(16)`, `rs(32)`, `rs(64)`;
- **Segurança de carga**: `torch.load(weights_only=True)` por padrão.

## Reprodutibilidade
Manifesto inclui:
- versão Python/Torch;
- hash git (best effort);
- plataforma;
- seed inicial do Torch.

## Observabilidade
Telemetria pode ser incorporada no próprio container para correlacionar:
- estado interno salvo,
- métricas agregadas,
- eventos recentes de execução.

Isso reduz risco de perda de contexto em incidentes e acelera análise de regressões.
