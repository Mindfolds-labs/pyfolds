# FOLD Binary Specification (`.fold/.mind`)

## 1. Objetivo
Especificar o layout binário completo do container `.fold/.mind` para garantir interoperabilidade, validação defensiva e resistência básica a ataques de negação de serviço (DoS).

> Fonte normativa de implementação atual: `src/pyfolds/serialization/foldio.py`.

---

## 2. Convenções
- **Endianness:** `big-endian` para todos os campos binários (`struct` com prefixo `>`).
- **Unidades:** offsets e tamanhos em **bytes**.
- **Codificação textual:** UTF-8 para JSON.

### 2.1 Constantes de segurança (anti-DoS)
- `MAX_INDEX_SIZE = 100 * 1024 * 1024` (100 MiB) — limite rígido para o índice JSON.
- `MAX_CHUNK_SIZE = 1 * 1024 * 1024 * 1024` (1 GiB) — limite operacional recomendado por chunk (comprimido e descomprimido) para leitores em ambiente não-confiável.

> `MAX_INDEX_SIZE` está aplicado no leitor atual. `MAX_CHUNK_SIZE` é requisito de política para hardening adicional e para implementações externas compatíveis.

---

## 3. Layout físico do arquivo

```
+----------------------+  offset 0
| Header fixo          |  28 bytes
+----------------------+  offset = index_off
| Região de chunks     |  repetição [chunk_header + payload + ecc]
| ...                  |
+----------------------+  offset = index_off
| Índice JSON          |  index_len bytes
+----------------------+  EOF
```

### 3.1 Header global (`HEADER_FMT = ">8sIQQ"`)
Tamanho fixo: **28 bytes**.

| Campo       | Tipo binário | Offset | Tamanho | Descrição |
|-------------|--------------|--------|---------|-----------|
| `magic`     | `8s`         | 0      | 8       | Assinatura fixa `b"FOLDv1\0\0"`. |
| `header_len`| `uint32`     | 8      | 4       | Deve ser `28`. |
| `index_off` | `uint64`     | 12     | 8       | Offset absoluto do índice JSON. |
| `index_len` | `uint64`     | 20     | 8       | Tamanho do índice JSON em bytes. |

### 3.2 Chunk (`CHUNK_HDR_FMT = ">4sIQQII"`)
Cada chunk começa com header fixo de **32 bytes**.

| Campo        | Tipo binário | Offset relativo ao chunk | Tamanho | Descrição |
|--------------|--------------|--------------------------|---------|-----------|
| `ctype`      | `4s`         | 0                        | 4       | Código ASCII de 4 bytes (ex.: `TSAV`, `JSON`, `NPZ0`). |
| `flags`      | `uint32`     | 4                        | 4       | Flags de compressão (`0=none`, `1=zstd`). |
| `uncomp_len` | `uint64`     | 8                        | 8       | Tamanho esperado após descompressão. |
| `comp_len`   | `uint64`     | 16                       | 8       | Tamanho da carga comprimida. |
| `crc32c`     | `uint32`     | 24                       | 4       | CRC32C da carga comprimida. |
| `ecc_len`    | `uint32`     | 28                       | 4       | Tamanho do bloco ECC anexado. |

Após o header:
1. `comp_payload` com `comp_len` bytes.
2. `ecc_payload` com `ecc_len` bytes.

---

## 4. Índice JSON (trailer)
O índice JSON fica ao final do arquivo, no offset `index_off`, e possui `index_len` bytes.

Estrutura canônica:
- `format`: `"fold"`
- `version`: versão de formato (atual `"1.2.0"`)
- `created_at_unix`: timestamp UNIX
- `metadata`: metadados de alto nível (inclui `chunk_hashes` e `manifest_hash`)
- `chunks`: lista de descritores por chunk

Cada item em `chunks` inclui, no mínimo:
- `name`, `ctype`, `flags`, `offset`, `header_len`, `comp_len`, `uncomp_len`, `crc32c`, `sha256`, `ecc_algo`, `ecc_len`.

---

## 5. Regras de escrita
Para cada chunk:
1. Gerar payload bruto.
2. Comprimir (opcional).
3. Calcular `crc32c` e `sha256` sobre o payload comprimido.
4. Aplicar ECC opcional sobre o payload comprimido.
5. Persistir: `chunk_header + comp_payload + ecc_payload`.
6. Registrar entrada correspondente no índice.

Finalize:
1. Serializar e gravar índice JSON no final do arquivo.
2. Regravar header global com `index_off` e `index_len` finais.

---

## 6. Regras de validação multicamada (leitura)
Ordem recomendada para leitores não-confiáveis:

1. **Validação estrutural do arquivo**
   - arquivo legível;
   - header completo (28 bytes);
   - `magic` correto;
   - `header_len == 28`;
   - `index_off >= header_len`.

2. **Limites anti-DoS**
   - `index_len <= MAX_INDEX_SIZE`;
   - para cada chunk: `comp_len <= MAX_CHUNK_SIZE` e `uncomp_len <= MAX_CHUNK_SIZE`.

3. **Validação de bounds/EOF**
   - toda leitura deve garantir `offset + length <= file_size`.

4. **Validação de integridade por chunk**
   - aplicar decode ECC (quando presente);
   - comparar `crc32c`;
   - comparar `sha256` do índice;
   - validar hash hierárquico em `metadata.chunk_hashes`.

5. **Validação de manifesto/metadados**
   - recomputar e comparar `metadata.manifest_hash` (aviso ou erro conforme política).

6. **Validação semântica de payload**
   - descompressão conforme `flags`;
   - tamanho descomprimido deve casar com `uncomp_len`;
   - desserialização de objetos em modo seguro (ex.: `torch.load(..., weights_only=True)`).

---

## 7. Chunks padrão (v1.2)
- `torch_state` (`TSAV`) — estado de treino para retomada.
- `llm_manifest` (`JSON`) — manifesto para auditoria/tooling.
- `metrics` (`JSON`) — métricas agregadas.
- `history` (`JSON`, opcional) — histórico de accumulator.
- `telemetry` (`JSON`, opcional) — snapshot de eventos/estatísticas.
- `nuclear_arrays` (`NPZ0`, opcional) — arrays científicos parciais.

---

## 8. Compatibilidade e extensibilidade
- Extensões `.fold` e `.mind` compartilham layout físico.
- Novos chunks são permitidos sem quebrar leitores antigos (desde que preservem header e índice).
- Recomendado manter versionamento semântico no campo `version` do índice.
