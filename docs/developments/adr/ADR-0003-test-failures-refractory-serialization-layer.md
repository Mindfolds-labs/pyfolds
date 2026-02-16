# ADR-0003 — Correções de regressão em refratário, serialização e API de camada

## Status
Accepted

## Contexto
A execução da suíte indicou 6 falhas e 1 teste pulado:
- falha no comportamento do refratário relativo (`blocked` inconsistente com semântica documentada);
- falhas de serialização por `NameError` e fallback de checksum incorreto para CRC32C;
- falha de API por atributo legado `neuron_class` ainda exposto;
- um teste ECC pulado quando `reedsolo` não está disponível no ambiente.

Esses problemas impediam confiança no comportamento do neurônio avançado, no formato `.fold/.mind` e na estabilidade de API.

## Decisão
1. **Refratário relativo sem bloqueio direto**
   - Ajustar `RefractoryMixin` para bloquear apenas na janela absoluta.
   - A janela relativa passa a atuar apenas via `theta_boost`.

2. **CRC32C correto sem dependência obrigatória**
   - Manter uso preferencial de `google-crc32c` quando disponível.
   - Substituir fallback incorreto (CRC32/zlib) por implementação pura de CRC32C (Castagnoli), evitando divergência do vetor conhecido.

3. **Remoção da exposição do atributo legado**
   - Preservar `neuron_cls` como API oficial em `MPJRDLayer`.
   - Remover `self.neuron_class` para eliminar ambiguidade e alinhar com testes de contrato.

4. **Teste ECC sem skip obrigatório por dependência externa**
   - Ajustar o teste de roundtrip ECC para sempre rodar:
     - com `ReedSolomonECC` quando `reedsolo` existir;
     - com `NoECC` quando não existir.
   - Manter teste de mapeamento que já valida comportamento condicional da dependência.

## Consequências
### Positivas
- Semântica do refratário fica consistente com documentação e assertivas principais.
- Integridade do container deixa de depender de fallback incorreto.
- API de camada fica mais limpa e previsível.
- Suíte reduz sensibilidade a variações de ambiente (menos testes pulados).

### Trade-offs
- Remoção de `neuron_class` pode afetar código externo que ainda dependia do atributo.
- Fallback puro de CRC32C é mais lento que implementação nativa acelerada.

## Validação
- Reexecução da suíte após as mudanças.
- Verificação explícita do vetor padrão `crc32c("123456789") == 0xE3069283`.
