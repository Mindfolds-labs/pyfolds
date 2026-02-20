# EXEC-012 — Execução da validação de criptografia e telemetria

## Checklist de execução
- [x] Instalar dependência opcional de criptografia.
- [x] Executar teste de assinatura digital `foldio`.
- [x] Executar teste de performance de overhead de telemetria.
- [x] Atualizar HUB e cards com os artefatos desta execução.

## Comandos
```bash
python -m pip install cryptography
pytest tests/unit/serialization/test_foldio.py::test_fold_signature_roundtrip_if_cryptography_available -v
pytest tests/performance/test_telemetry_overhead.py -v -s -m "slow and performance"
```

## Observações
- O comando de performance sem `-m "slow and performance"` fica deselectionado por configuração padrão do `pytest.ini`.
- Telemetria permaneceu estável sem falhas de invariantes (`spikes` e `u` válidos durante a medição).
