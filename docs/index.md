# PyFolds Documentation

![PyFolds banner](_static/brand/pyfolds-readme-banner.svg)

Documentação técnica estruturada em seis eixos principais.

## Conteúdo técnico

```{toctree}
:maxdepth: 2
:caption: Navegação principal

architecture/index
science/index
mechanisms/index
development/index
sheer-audit/index
governance/adr/index
```

## Referências/Legado

- [Portal do usuário (legado de navegação)](user/index.md)
- [API reference (legado de navegação)](api/index.md)
- [Governança geral (visão ampla)](governance/index.md)
- [README da documentação](README.md)
- [Portal interno de prompts](development/prompts.md)

## Mapa do Código

Atualizado automaticamente pelo Sheer Audit no merge para `main`.

<!-- SHEER-CODEMAP:START -->
- Repositório: `pyfolds`
- Arquivos Python: `143`
- Símbolos: `1084`


- Arquivo: `src/pyfolds/__init__.py`
- Imports:
  - `mod:src.advanced`
  - `mod:src.bridge`
  - `mod:src.core.base`
  - `mod:src.core.config`
  - `mod:src.core.factory`
  - `mod:src.core.neuron`
  - `mod:src.core.neuron_v2`
  - `mod:src.layers`
  - `mod:src.monitoring`
  - `mod:src.network`
  - `mod:src.serialization`
  - `mod:src.telemetry`
  - `mod:src.utils.context`
  - `mod:src.utils.types`
  - `mod:src.wave`
  - `mod:typing`
  - `mod:warnings`
- Funções:
  - `__getattr__(name)`

- Arquivo: `src/pyfolds/advanced/__init__.py`
- Imports:
  - `mod:logging`
  - `mod:src.core.neuron`
  - `mod:src.layers.layer`
  - `mod:src.pyfolds.adaptation`
  - `mod:src.pyfolds.backprop`
  - `mod:src.pyfolds.inhibition`
  - `mod:src.pyfolds.refractory`
  - `mod:src.pyfolds.short_term`
  - `mod:src.pyfolds.stdp`
  - `mod:src.utils.logging`
<!-- SHEER-CODEMAP:END -->

Consulte também: `docs/sheer-audit/sheerdocs/code_map.md`.
