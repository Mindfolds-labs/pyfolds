# Hardening Report — PyFolds v2.1.1 Plan

## Escopo aplicado

1. **Modernização UTC (Python 3.12+)**
   - Substituição de `datetime.utcnow()` por `datetime.now(UTC)` em logging.

2. **Serialização segura (`weights_only=True`)**
   - Registro explícito de safe globals no carregamento.
   - Normalização de metadados para tipos primitivos seguros.

3. **Dependências de desenvolvimento e cobertura**
   - Inclusão de `cryptography` e `pytest-cov` em `project.optional-dependencies.dev`.
   - Execução do teste de assinatura Ed25519 e suíte com cobertura.

4. **Silenciamento de warning de backward hook com saída em dict**
   - Remoção do `register_full_backward_hook` no neurônio.
   - Migração para hooks de gradiente por parâmetro (`param.register_hook`).

5. **Integridade (bitflip/paridade + simulação ECC-like)**
   - Novos testes para corrupção em rajada (multi-byte bitflip).
   - Hardening em `MPJRDSynapse` para saneamento de buffers corrompidos (NaN/Inf/out-of-range).

## Evidências de validação

- `PYTHONPATH=src pytest tests/unit/serialization/test_foldio.py::test_fold_signature_roundtrip_if_cryptography_available -v`
- `PYTHONPATH=src pytest tests/ -v --cov=src/pyfolds`
- `PYTHONPATH=src python tools/verify_hardening.py`

## Resultado

- Suíte completa: **319 passed**.
- Cobertura total: **79%**.
- Teste de assinatura criptográfica: **executado e aprovado**.
- Novos cenários de corrupção (ECC-like) validados.
