# FOLD Binary Specification (`.fold` / `.mind`)

## 1) Escopo e objetivos

Este documento define o layout binário canônico do container `.fold/.mind` implementado em `src/pyfolds/serialization/foldio.py`.

Objetivos do formato:
- permitir **leitura parcial** sem desserializar tudo;
- fornecer **integridade por chunk** (CRC32C + SHA-256);
- suportar **ECC opcional por chunk**;
- manter compatibilidade com serialização científica baseada em PyTorch.

---

## 2) Convenções de codificação binária

- **Endianness:** big-endian (`>` no `struct.pack/unpack`).
- **Tipos inteiros:**
  - `I` = `uint32` (4 bytes)
  - `Q` = `uint64` (8 bytes)
- **Strings fixas:** ASCII/bytes fixos.
- **Offsets:** sempre relativos ao início do arquivo.

Constantes de referência:
- `MAGIC = b"FOLDv1\0\0"`
- `HEADER_FMT = ">8sIQQ"`
- `CHUNK_HDR_FMT = ">4sIQQII"`
- `FLAG_COMP_NONE = 0`
- `FLAG_COMP_ZSTD = 1`

---

## 3) Layout físico do arquivo

Ordem de escrita no arquivo:

1. **Header global** (tamanho fixo)
2. **Sequência de chunks** (header + payload comprimido + ECC opcional)
3. **Índice JSON final**
4. Reescrita do header com `index_off` e `index_len` finais

Representação:

```text
+----------------------+  offset 0
| Global Header        |
+----------------------+  offset = header_len
| Chunk #0             |
+----------------------+  ...
| Chunk #1             |
+----------------------+  ...
| ...                  |
+----------------------+  index_off
| JSON Index           |
+----------------------+  EOF
```

---

## 4) Header global (`>8sIQQ`, 28 bytes)

| Campo       | Offset | Tamanho | Tipo   | Descrição |
|-------------|--------|---------|--------|-----------|
| `magic`     | 0      | 8       | `8s`   | Assinatura fixa `FOLDv1\0\0` |
| `header_len`| 8      | 4       | `I`    | Tamanho do header (deve ser 28 na v1.x) |
| `index_off` | 12     | 8       | `Q`    | Offset absoluto do índice JSON |
| `index_len` | 20     | 8       | `Q`    | Tamanho em bytes do índice JSON |

Regras:
- `magic` deve bater exatamente;
- `header_len` deve ser igual a `struct.calcsize(HEADER_FMT)`;
- `index_off >= header_len`;
- `index_len <= MAX_INDEX_SIZE`.

---

## 5) Header de chunk (`>4sIQQII`, 32 bytes)

Cada chunk começa com um cabeçalho binário fixo:

| Campo        | Offset (relativo ao chunk) | Tamanho | Tipo | Descrição |
|--------------|----------------------------|---------|------|-----------|
| `ctype4`     | 0                          | 4       | `4s` | Tipo lógico em 4 chars (ex.: `TSAV`, `JSON`) |
| `flags`      | 4                          | 4       | `I`  | Flags de compressão (`0` none, `1` zstd) |
| `uncomp_len` | 8                          | 8       | `Q`  | Tamanho original do payload |
| `comp_len`   | 16                         | 8       | `Q`  | Tamanho do payload comprimido |
| `crc32c`     | 24                         | 4       | `I`  | CRC32C do payload **comprimido** |
| `ecc_len`    | 28                         | 4       | `I`  | Tamanho dos bytes ECC anexados |

Layout completo do chunk no arquivo:

```text
chunk_offset
  +0   : ChunkHeader (32 bytes)
  +32  : comp_bytes (comp_len)
  +32+comp_len : ecc_bytes (ecc_len)
```

Observação importante:
- CRC/SHA/ECC são calculados sobre `comp_bytes` (payload pós-compressão, pré-descompressão).

---

## 6) Índice JSON final

O índice é um JSON UTF-8 com estrutura:

- `format`: string (`"fold"`)
- `version`: versão do formato (ex.: `"1.2.0"`)
- `created_at_unix`: timestamp float
- `metadata`: metadados globais (inclui `chunk_hashes` e `manifest_hash`)
- `chunks`: array de entradas, uma por chunk

Campos esperados por item de `chunks`:
- `name`
- `ctype`
- `flags`
- `offset`
- `header_len`
- `comp_len`
- `uncomp_len`
- `crc32c`
- `sha256`
- `ecc_algo`
- `ecc_len`

A posição (`offset`) de cada chunk aponta para o início do header binário de chunk.

---

## 7) Algoritmo de leitura (referência)

1. Abrir arquivo (com ou sem `mmap`).
2. Ler e validar header global.
3. Ler `index_len` bytes em `index_off`.
4. Fazer parse UTF-8 + JSON.
5. Resolver chunk por `name` no índice.
6. Ler header binário do chunk em `chunk.offset`.
7. Ler `comp_bytes` e `ecc_bytes` usando `comp_len`/`ecc_len`.
8. Se `ecc_algo != none`, aplicar decodificação ECC.
9. Se `verify=True`:
   - validar CRC32C de `comp_bytes`;
   - validar SHA-256 de `comp_bytes` contra `chunk.sha256`;
   - validar consistência hierárquica (`metadata.chunk_hashes`).
10. Descomprimir (`flags`).
11. Validar `len(raw) == uncomp_len`.
12. Retornar bytes crus ou decodificar (JSON/Torch/etc).

Ordem de segurança na leitura: **ECC -> integridade -> descompressão -> decode de alto nível**.

---

## 8) Regras de validação obrigatórias

### 8.1 Header e index
- rejeitar magic inválido;
- rejeitar `header_len` diferente do valor canônico;
- rejeitar `index_off < header_len`;
- rejeitar `index_len > MAX_INDEX_SIZE`;
- rejeitar EOF/truncamento em qualquer leitura de range.

### 8.2 Chunk bounds
Para cada chunk recuperado do índice:
- `offset >= header_len`;
- `offset + chunk_header_size` deve caber no arquivo;
- `offset + chunk_header_size + comp_len + ecc_len` deve caber no arquivo.

### 8.3 Integridade e consistência
- CRC32C deve bater com `crc32c` do header binário do chunk;
- SHA-256 deve bater com `sha256` do índice;
- `metadata.chunk_hashes[name]` (se presente) deve bater com `sha256`.

### 8.4 Pós-descompressão
- tamanho final deve ser exatamente `uncomp_len`.

---

## 9) Limites anti-DoS

### 9.1 Limite vigente: `MAX_INDEX_SIZE`

`MAX_INDEX_SIZE = 100 * 1024 * 1024` (100 MiB) já é usado para impedir alocação excessiva ao carregar o índice JSON.

Justificativa:
- índice é estrutura de metadados, tipicamente pequena;
- limite rígido evita arquivos maliciosos com `index_len` gigante;
- reduz risco de OOM e latência extrema na desserialização inicial.

### 9.2 Proposta: `MAX_CHUNK_SIZE`

**Proposta de hardening:** adicionar limite explícito para `comp_len` e/ou `uncomp_len` por chunk.

Sugestão pragmática inicial:
- `MAX_CHUNK_SIZE = 1 * 1024 * 1024 * 1024` (1 GiB) por chunk,
- com possibilidade de override controlado por parâmetro avançado.

Justificativa técnica:
- protege contra chunks gigantes que forçam leitura/alocação desnecessária;
- reduz superfície de ataques de descompressão e abuso de memória;
- mantém compatibilidade com checkpoints reais (que tendem a ser fragmentáveis em múltiplos chunks).

Regra recomendada ao implementar:
- falhar cedo quando `comp_len > MAX_CHUNK_SIZE` **ou** `uncomp_len > MAX_CHUNK_SIZE`.

---

## 10) Compatibilidade e versionamento

- v1.x assume header de 28 bytes e chunk header de 32 bytes.
- Mudanças incompatíveis devem incrementar major (v2.x) e/ou magic.
- Extensão `.mind` é semântica de produto; layout físico permanece idêntico ao `.fold`.

---

## 11) Checklist rápido de conformidade

- [ ] Endianness big-endian em todos os structs binários.
- [ ] `MAGIC` validado byte a byte.
- [ ] `index_len` limitado por `MAX_INDEX_SIZE`.
- [ ] Bounds-check de todo read-at (`offset`, `length`).
- [ ] Verificação CRC32C + SHA-256 antes de descompressão.
- [ ] Validação de `uncomp_len` após descompressão.
- [ ] (Recomendado) `MAX_CHUNK_SIZE` aplicado para `comp_len`/`uncomp_len`.

