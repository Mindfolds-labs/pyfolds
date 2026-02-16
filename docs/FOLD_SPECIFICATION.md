# FOLD Binary Specification (`.fold` / `.mind`)

> Fonte normativa: implementação em `src/pyfolds/serialization/foldio.py`.

## 1. Endianness e tipos

Todos os campos binários usam **big-endian** (`>` no `struct`).

- `u32`: inteiro sem sinal de 4 bytes.
- `u64`: inteiro sem sinal de 8 bytes.
- `bytes[n]`: sequência de bytes sem terminador.

## 2. Layout físico do arquivo

Formato global:

1. **Header fixo** (`HEADER_FMT = ">8sIQQ"`, 28 bytes)
2. **Região de chunks** (0..N chunks, cada um com cabeçalho + payload comprimido + ECC)
3. **Índice JSON** UTF-8 no final, apontado pelo header

### 2.1 Header (`offset 0`)

| Offset | Tamanho | Tipo | Campo | Descrição |
|---:|---:|---|---|---|
| 0 | 8 | `8s` | `magic` | Valor fixo `b"FOLDv1\0\0"` |
| 8 | 4 | `u32` | `header_len` | Tamanho do header; deve ser `28` |
| 12 | 8 | `u64` | `index_off` | Offset absoluto do índice JSON |
| 20 | 8 | `u64` | `index_len` | Tamanho em bytes do índice JSON |

Validações obrigatórias na leitura:
- `magic` deve ser exato.
- `header_len == 28`.
- `index_off >= header_len`.
- `index_len <= MAX_INDEX_SIZE`.

## 3. Chunk layout

Cada chunk inicia em `chunk.offset` (registrado no índice), com cabeçalho fixo `CHUNK_HDR_FMT = ">4sIQQII"` (32 bytes):

| Offset relativo | Tamanho | Tipo | Campo | Descrição |
|---:|---:|---|---|---|
| 0 | 4 | `4s` | `ctype` | tipo físico (`TSAV`, `JSON`, `NPZ0`, …) |
| 4 | 4 | `u32` | `flags` | compressão (`0=none`, `1=zstd`) |
| 8 | 8 | `u64` | `uncomp_len` | tamanho esperado após descompressão |
| 16 | 8 | `u64` | `comp_len` | tamanho do payload comprimido |
| 24 | 4 | `u32` | `crc32c` | checksum CRC32C do payload comprimido |
| 28 | 4 | `u32` | `ecc_len` | tamanho do bloco ECC armazenado após `comp` |

Após os 32 bytes do cabeçalho:
- `comp[comp_len]`
- `ecc[ecc_len]` (opcional)

### 3.1 Limites

- `MAX_INDEX_SIZE = 100 * 1024 * 1024` (100 MiB)
- `MAX_CHUNK_SIZE = 2 * 1024 * 1024 * 1024` (2 GiB)

Aplicação dos limites:
- **Escrita**: rejeita payload bruto (`uncomp_len`) e comprimido (`comp_len`) acima de `MAX_CHUNK_SIZE`.
- **Leitura**: rejeita `uncomp_len`, `comp_len` e `ecc_len` acima de `MAX_CHUNK_SIZE`.

## 4. Índice JSON

O índice é serializado em UTF-8 e contém:

- `format` (ex.: `"fold"`)
- `version` (ex.: `"1.2.0"`)
- `created_at_unix`
- `metadata` (inclui `chunk_hashes` e `manifest_hash`)
- `chunks[]` com metadados de cada chunk:
  - `name`, `ctype`, `flags`, `offset`, `header_len`, `comp_len`, `uncomp_len`,
    `crc32c`, `sha256`, `ecc_algo`, `ecc_len`

## 5. Pipeline de escrita (normativo)

Para cada chunk:
1. Serializar payload lógico em bytes.
2. Comprimir (`none` ou `zstd`).
3. Calcular `crc32c` e `sha256` sobre o conteúdo comprimido.
4. Gerar bytes ECC opcionais sobre o conteúdo comprimido.
5. Persistir `[chunk_header | comp | ecc]`.
6. Registrar entrada em `chunks[]`.

Finalização:
1. Montar `metadata.chunk_hashes` com hashes por nome de chunk.
2. Calcular `metadata.manifest_hash` (SHA-256 canônico do metadata sem `manifest_hash`).
3. Escrever índice JSON no fim do arquivo.
4. Reescrever header com `index_off/index_len` finais.

## 6. Algoritmo de leitura (normativo)

1. Abrir arquivo (com ou sem `mmap`).
2. Ler 28 bytes do header e validar campos estruturais.
3. Ler e desserializar índice (`index_off`, `index_len`).
4. Para ler um chunk por nome:
   1. localizar metadado no índice;
   2. ler cabeçalho binário de 32 bytes no `offset`;
   3. validar limites de `uncomp_len`, `comp_len`, `ecc_len`;
   4. ler bytes `comp` e `ecc`;
   5. decodificar ECC (se `ecc_algo != none`);
   6. validar `crc32c(comp)` e `sha256(comp)`;
   7. validar hash hierárquico (`metadata.chunk_hashes[name]`);
   8. validar `manifest_hash` (emite warning se divergente);
   9. descomprimir (`flags`);
   10. validar `len(raw) == uncomp_len`.

## 7. Segurança

- `torch_state` é carregado com `torch.load(..., weights_only=True)` por padrão.
- Há modo explícito confiável (`trusted_torch_payload=True`) para `torch.load` sem restrição.
- Leitura defensiva evita EOF/offset inválido e índices/chunks excessivos.

## 8. `.fold` vs `.mind`

O formato físico é idêntico.
`is_mind` é definido semanticamente pela presença de chunks `ai_graph` ou `ai_vectors`.
