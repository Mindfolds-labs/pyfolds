# Noetic no PyFolds

O `NoeticCore` integra memória por engrams, temporalidade circadiana e especialização por área.

## Componentes
- `Engram` e `EngramBank`: criação, busca por ressonância, replay e pruning.
- `SpecializationEngine`: hierarquia de conhecimento e síntese interdisciplinar.
- `NoeticCore`: ciclo de vida completo (learn/query/sleep/save/load).

## Conceitos-chave
- **Engram**: traço distribuído com frequência, fase e timestamp.
- **Sono**: `replay()` + `consolidate(pruning=True)`.
- **Especialização**: níveis harmônicos por área (`specialize`).
- **Síntese**: criação de engram relacional entre áreas (`synthesize`).

## Exemplo
```python
from pyfolds.core.config import MPJRDConfig
from pyfolds.advanced.noetic_model import NoeticCore

cfg = MPJRDConfig(wave_enabled=True, max_engrams=100_000)
noetic = NoeticCore(cfg)
noetic.specialization.define_area("fisica", "Física", base_frequency=50.0)
noetic.learn("gravidade", area="fisica", importance=0.9)
print(noetic.query("gravidade", area="fisica"))
noetic.sleep()
```
