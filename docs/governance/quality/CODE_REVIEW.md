# Guia de Revisão de Código - PyFolds

## Padrões de Qualidade

### 1. PEP 8 (Estilo de Código)
- [ ] Linhas com no máximo 79 caracteres
- [ ] Importações em ordem: padrão → terceiros → locais
- [ ] Nomes de classes em CamelCase
- [ ] Nomes de funções/métodos em snake_case

### 2. Type Hints
- [ ] Todas as funções públicas têm type hints
- [ ] Tipos complexos usam `from typing import ...`
- [ ] Retornos opcionais são `Optional[Tipo]`

```python
def forward(
    self,
    x: torch.Tensor,
    reward: Optional[float] = None,
    mode: Optional[LearningMode] = None,
) -> Dict[str, torch.Tensor]:
    ...
```

### 3. Docstrings (Formato Google)

```python
def apply_plasticity(self, dt: float = 1.0, reward: Optional[float] = None) -> None:
    """Aplica plasticidade baseada em estatísticas acumuladas.

    Args:
        dt: Passo de tempo em ms (padrão: 1.0)
        reward: Sinal de recompensa externo (opcional)

    Returns:
        None

    Raises:
        ValueError: Se reward for None no modo 'external'
    """
```

### 4. Testes Unitários
- Cobertura mínima: 80%.
- Testes para casos de borda (`N=0`, `N=31`).
- Testes para modos (`ONLINE`, `BATCH`, `SLEEP`, `INFERENCE`).

### 5. Checklist de Revisão

#### Funcionalidade
- [ ] O código faz o que a documentação diz?
- [ ] Os casos de erro são tratados adequadamente?

#### Eficiência
- [ ] Operações vetorizadas sem loops desnecessários?
- [ ] Caching implementado onde apropriado?
- [ ] Uso correto de buffers vs parâmetros?

#### Manutenibilidade
- [ ] Código legível e auto-documentado?
- [ ] Constantes mágicas evitadas?
- [ ] Configurações extraídas de `MPJRDConfig`?

#### Compatibilidade
- [ ] Funciona em CPU e CUDA?
- [ ] Compatível com versões anteriores?
