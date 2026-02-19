# Contrato de API pública v2 (`pyfolds`)

Este documento define o contrato da superfície pública **v2** exposta por `pyfolds` no módulo raiz (`src/pyfolds/__init__.py`) e os aliases de compatibilidade mantidos para migração da API v1.

## Símbolos públicos v2 (canônicos)

Os símbolos abaixo são a superfície recomendada para novos usuários e para evolução futura:

- `NeuronConfig` → configuração do neurônio (canônico da v2; aponta para a implementação atual de configuração).
- `NeuronState` → contrato de tipo para o estado retornado por chamadas `forward` e similares.
- `MPJRDNeuron` → classe de neurônio base.
- `MPJRDNeuronV2` → variante de neurônio com integração cooperativa.
- `AdaptiveNeuronLayer` → camada adaptativa de neurônios.
- `SpikingNetwork` → rede de camadas neurais spike-based.
- `MPJRDWaveLayer`, `MPJRDWaveNetwork` e `NetworkBuilder` → superfície de construção de topologias em modo wave/rede.

## Compatibilidade v1 (aliases de migração)

Para permitir upgrade incremental no mesmo release, a API mantém aliases legados com depreciação:

- `MPJRDConfig` → alias para `NeuronConfig` (**deprecated**).
- `MPJRDLayer` → alias para `AdaptiveNeuronLayer` (**deprecated**).
- `MPJRDNetwork` → alias para `SpikingNetwork` (**deprecated**).

### Política de depreciação

- O acesso aos símbolos legados deve emitir `DeprecationWarning`.
- Código novo deve usar os nomes canônicos v2.
- Os aliases legados existem apenas como ponte de migração entre v1 e v2.

### Cronograma de remoção dos aliases v1

| Marco | Regra | Resultado esperado |
| --- | --- | --- |
| `1.x` (janela de migração) | Aliases v1 continuam disponíveis com `DeprecationWarning`. | Compatibilidade para upgrade gradual. |
| `2.0.0` (versão-limite) | Aliases v1 são removidos da superfície pública. | Importações legadas passam a falhar, exigindo nomes v2. |

Critérios objetivos para a remoção:

1. **Ciclos mínimos:** manter aliases por, no mínimo, a janela de migração da série `1.x` antes da remoção.
2. **Aviso em changelog:** registrar explicitamente em `CHANGELOG.md` que aliases v1 serão removidos em `2.0.0`.
3. **Estratégia de migração:** orientar substituição direta de símbolos (`MPJRDConfig` → `NeuronConfig`, `MPJRDLayer` → `AdaptiveNeuronLayer`, `MPJRDNetwork` → `SpikingNetwork`) em documentação pública e notas de release.

## Garantias de release

1. Importar símbolos v2 diretamente de `pyfolds` deve funcionar.
2. Importar símbolos v1 deve continuar funcional no mesmo release, com `DeprecationWarning`.
3. O conjunto de `__all__` em `pyfolds` deve listar tanto os nomes canônicos v2 quanto aliases v1 enquanto a janela de migração estiver ativa.
4. Enquanto `pyfolds.__version__ < 2.0.0`, acesso a aliases v1 deve seguir emitindo `DeprecationWarning` (coberto por teste de contrato).

## Referências de implementação e testes

- Implementação da superfície pública: `src/pyfolds/__init__.py`
- Testes de contrato de import: `tests/unit/test_public_import_surface.py`
