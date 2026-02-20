# Stable Core Architecture (Freeze v2.1.1)

## Objetivo
Este documento fixa os contratos técnicos estáveis do núcleo PyFolds no Release Candidate v2.1.1, garantindo evolução sem quebra do legado MPJRD.

## 1) Identidade de Configuração: MPJRDConfig como fonte única

- A classe base científica e serializável do projeto é `MPJRDConfig`.
- `NeuronConfig` é um alias operacional/comercial para a mesma classe:

```python
NeuronConfig = MPJRDConfig
```

### Implicação arquitetural
- Não existem duas árvores de configuração.
- Qualquer código legado que serializa/desserializa `MPJRDConfig` continua funcional.
- Novos consumidores podem adotar o nome `NeuronConfig` sem bifurcar comportamento.

## 2) Contrato de Bridge/Dispatcher em mão dupla

O `MindDispatcher` emite payload canônico para integrações futuras **e** preserva chaves legadas no mesmo evento.

### Chaves canônicas (v2)
- `layer`
- `ts`
- `payload`

### Chaves de compatibilidade (legado)
- `layer_id`
- `timestamp`

### Garantia
- `layer_id` espelha `layer`.
- `timestamp` espelha `ts`.
- Consumidores antigos e novos podem coexistir sem adaptadores obrigatórios.

## 3) Blindagem de checkpoint (torch.load / safe globals)

O módulo de checkpoint registra explicitamente ambos os nomes de configuração em `safe_globals`:

- `MPJRDConfig`
- `NeuronConfig`

Isso reduz fricção de carregamento seguro (`weights_only=True`) em ambientes mistos e mantém a compatibilidade de objetos serializados.

## 4) Supressão seletiva de ruído de terceiros

Durante carregamento de checkpoint, apenas `UserWarning` do PyTorch relacionados a `dict`/`dictionary` são silenciados de forma localizada.

### Princípio de segurança
- **Não** há supressão global de warnings.
- Erros de lógica, integridade e incompatibilidade de shape continuam visíveis e bloqueantes.

## 5) Política de estabilidade para RC

- Não remover `MPJRDConfig`.
- Não quebrar chaves históricas de payload em fluxo crítico.
- Evolução via alias e dual-contract, nunca por substituição abrupta.

## 6) Critério de aceite operacional

- Suite de testes executada com:

```bash
PYTHONPATH=src pytest tests/ -v
```

- Resultado esperado no freeze: regressão funcional zero nos contratos legados críticos.
