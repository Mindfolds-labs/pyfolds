# ADR-045 — Checkpoints seguros com safetensors, flush lazy e validação de shape

- **Status:** Ativo
- **Data:** 2026-02-20
- **Decisores:** Engenharia de Serialização, Runtime e Confiabilidade
- **Contexto:** O ciclo de auditoria técnica identificou três eixos de melhoria para checkpoints e logging: blindagem contra desserialização insegura, redução de I/O síncrono em logging circular e validação estrutural antes de injetar pesos no modelo.

## Contexto

O `VersionedCheckpoint` já possuía validação de integridade por hash/HMAC e carregamento com `weights_only=True` quando suportado. Porém, ainda havia dependência de payload pickle para checkpoints `.pt`, além de ausência de validação explícita de compatibilidade de shape antes do `load_state_dict`.

No logging, o `CircularBufferFileHandler` reescrevia o arquivo a cada `emit`, com custo O(n) por mensagem e impacto potencial em treinos longos.

## Decisão

Adotar as seguintes decisões arquiteturais para release `2.0.2`:

1. **Suporte nativo a `safetensors` em `VersionedCheckpoint`:**
   - `save(..., use_safetensors=True)` passa a salvar pesos em `.safetensors`.
   - Metadados e hash de integridade ficam em sidecar `*.safetensors.meta.json`.
   - `load()` aceita arquivos `.safetensors` com validações já existentes de integridade/versão.

2. **Validação de shape antes de restaurar estado:**
   - `VersionedCheckpoint.load(..., model=...)` valida `named_parameters()` vs `state_dict` carregado.
   - Mismatch gera `ValueError` explícito e bloqueia carga parcial silenciosa.

3. **Lazy flush em `CircularBufferFileHandler`:**
   - Escrita em disco por intervalo (`flush_interval_sec`) ou imediatamente em erro (`ERROR+`).
   - `close()` garante flush de buffer pendente.

## Consequências

### Positivas
- Redução de superfície de risco de desserialização com formato imutável para pesos.
- Menor pressão de I/O em logging contínuo.
- Falha rápida em incompatibilidades estruturais de checkpoint.

### Trade-offs
- Checkpoints em `safetensors` passam a usar sidecar de metadados.
- Novos parâmetros de configuração de logging exigem documentação e cobertura de testes.

## Implementação vinculada

- `src/pyfolds/serialization/versioned_checkpoint.py`
- `src/pyfolds/utils/logging.py`
- `tests/unit/core/test_monitoring_and_checkpoint.py`
- `tests/unit/utils/test_logging.py`

## Referências

- `docs/development/release_process.md`
- `CHANGELOG.md`
