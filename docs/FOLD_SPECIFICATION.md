# FOLD Specification (`.fold/.mind`) — v1.2.0

## 1. Escopo e objetivo

Este documento formaliza o formato binário do container `.fold/.mind` usado no PyFolds para checkpoints científicos, inspeção parcial, validação de integridade e recuperação operacional.

- `.fold` e `.mind` têm **layout físico idêntico**;
- `.mind` é uma convenção semântica quando o índice declara chunks de IA (`ai_graph` e/ou `ai_vectors`).

## 2. Convenções binárias

### 2.1 Endianness

Todos os campos binários usam **big-endian** (network byte order), conforme os formatos:

- Header do arquivo: `>8sIQQ`
- Header de chunk: `>4sIQQII`

### 2.2 Header principal (28 bytes)

| Offset (hex) | Tamanho | Tipo | Campo | Descrição |
|---|---:|---|---|---|
| `0x00` | 8 | `8s` | `magic` | Assinatura fixa `b"FOLDv1\\0\\0"` |
| `0x08` | 4 | `uint32` | `header_len` | Tamanho do header principal; na v1.2.0 = `28` |
| `0x0C` | 8 | `uint64` | `index_off` | Offset absoluto do índice JSON |
| `0x14` | 8 | `uint64` | `index_len` | Tamanho do índice JSON em bytes |

### 2.3 Header de chunk (32 bytes)

Cada chunk inicia em `chunk.offset` (registrado no índice), seguido por payload comprimido e bytes ECC opcionais.

| Offset relativo | Tamanho | Tipo | Campo | Descrição |
|---|---:|---|---|---|
| `+0x00` | 4 | `4s` | `ctype` | Tipo fixo de 4 bytes ASCII (`TSAV`, `JSON`, `NPZ0`, etc.) |
| `+0x04` | 4 | `uint32` | `flags` | Compressão (`0=none`, `1=zstd`) |
| `+0x08` | 8 | `uint64` | `uncomp_len` | Tamanho original (descomprimido) |
| `+0x10` | 8 | `uint64` | `comp_len` | Tamanho comprimido |
| `+0x18` | 4 | `uint32` | `crc32c` | CRC32C (Castagnoli) do payload comprimido |
| `+0x1C` | 4 | `uint32` | `ecc_len` | Tamanho do bloco ECC (0 se ausente) |

Layout físico completo por chunk:

```text
[ chunk_header(32) ][ compressed_payload(comp_len) ][ ecc_bytes(ecc_len) ]
```

### 2.4 Índice JSON (tail index)

O índice JSON é gravado no final do arquivo e seu endereço é referenciado no header principal (`index_off`, `index_len`). Estrutura mínima esperada:

- `format`: string (`"fold"`)
- `version`: string semântica (ex.: `"1.2.0"`)
- `created_at_unix`: `float`
- `metadata`: objeto (inclui `chunk_hashes` e `manifest_hash`)
- `chunks`: array de objetos com descritores por chunk:
  - `name`, `ctype`, `flags`, `offset`, `header_len`, `comp_len`, `uncomp_len`, `crc32c`, `sha256`, `ecc_algo`, `ecc_len`

## 3. Regras de validação e limites anti-DoS

A implementação de leitura **deve falhar de forma explícita** sob qualquer inconsistência estrutural.

### 3.1 Header e índice

1. `magic` deve ser exatamente `FOLDv1\0\0`.
2. `header_len` deve ser igual a `28` (`struct.calcsize(">8sIQQ")`).
3. `index_off >= header_len`.
4. `index_len <= MAX_INDEX_SIZE`.
5. `index_off + index_len` deve estar dentro do tamanho físico do arquivo.
6. Índice deve ser UTF-8 e JSON válido.

### 3.2 Limites anti-DoS

- `MAX_INDEX_SIZE = 100 * 1024 * 1024` (100 MiB).
- `MAX_CHUNK_SIZE = 1 * 1024 * 1024 * 1024` (1 GiB) para `comp_len` e `uncomp_len`.

Observação: `MAX_CHUNK_SIZE` é aplicado como limite de defesa para evitar alocação excessiva, offsets maliciosos e tentativas de amplificação de memória.

### 3.3 Integridade e consistência por chunk

Para cada chunk solicitado:

1. Ler header de chunk em `offset`.
2. Validar limites (`comp_len`, `uncomp_len`) contra `MAX_CHUNK_SIZE`.
3. Ler `comp` e `ecc_bytes` sem ultrapassar o tamanho do arquivo.
4. Se `ecc_algo != none`, decodificar ECC antes da validação criptográfica.
5. Validar `crc32c(comp)` contra `crc32c` do header.
6. Validar `sha256(comp)` contra `sha256` do índice.
7. Validar consistência opcional `metadata.chunk_hashes[name] == chunk.sha256`.
8. Descomprimir conforme `flags`.
9. Validar `len(raw) == uncomp_len`.

## 4. Algoritmo de leitura passo a passo

### 4.1 Procedimento normativo

1. Abrir arquivo em modo binário (`rb`) e opcionalmente mapear via `mmap`.
2. Ler 28 bytes iniciais (header principal).
3. Fazer `unpack('>8sIQQ')`.
4. Aplicar validações de header e limite de índice.
5. Ler `index_len` bytes a partir de `index_off`.
6. Decodificar UTF-8 e parsear JSON.
7. Para listar chunks, retornar `chunks[*].name`.
8. Para leitura de um chunk por nome:
   - localizar objeto no array `chunks`;
   - ler header do chunk em `chunk.offset` (`'>4sIQQII'`);
   - ler payload comprimido e ECC;
   - aplicar validações de integridade;
   - descomprimir (se necessário);
   - retornar bytes descomprimidos.
9. Para payload JSON: `json.loads(bytes.decode('utf-8'))`.
10. Para payload `torch_state`: carregar em modo seguro (`weights_only=True`) por padrão; modo confiável (`trusted`) somente sob decisão explícita.

## 5. Compatibilidade e governança

- Versão atual: `1.2.0`.
- Alterações incompatíveis exigem novo `magic` ou política formal de migração.
- Alterações compatíveis aditivas devem preservar leitura de versões anteriores.
- Decisões arquiteturais e operacionais relacionadas ao formato são registradas em `docs/adr/`.
