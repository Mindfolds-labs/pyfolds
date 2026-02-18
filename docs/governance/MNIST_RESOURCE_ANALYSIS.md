# Análise de custo de memória e processamento (MNIST + PyFolds)

## Configuração de referência

- Entrada MNIST: `1x28x28` (784 features)
- Exemplo de camada: `n_neurons=24`, `n_dendrites=4`, `n_synapses=8`
- Batch típico: 32
- Precisão: `float32`

## Estimativa de consumo

### Memória de ativações (aproximação)

- Tensor de entrada para camada MPJRD: `[B, N, D, S]`
- Elementos: `32 * 24 * 4 * 8 = 24_576`
- Memória bruta (fp32): `24_576 * 4 bytes ≈ 96 KB` por lote (apenas entrada da camada)
- Com gradientes + buffers intermediários + otimização Adam, o custo efetivo tende a múltiplos de 3x a 8x.

### Memória de parâmetros (aproximação)

- Projeção linear `784 -> (N*D*S = 768)`: `784*768 ≈ 602k` pesos + bias
- Cabeça `24 -> 10`: ~250 pesos
- Estados Adam: ~2x parâmetros treináveis

## Gargalos principais

1. **Projeção densa inicial** (`Linear(784, N*D*S)`), dominante em FLOPs.
2. **Atualização local por sinapse** no modo ONLINE (custos de plasticidade por lote).
3. **Telemetria/logging excessivos** em frequência alta (I/O de disco e lock contention).

## Otimizações possíveis

- Reduzir `N`, `D`, `S` em experimentos iniciais (escala progressiva).
- Usar `LearningMode.INFERENCE` fora da fase de atualização.
- Log em arquivo com frequência controlada (por época, não por iteração).
- Ajustar `num_workers=0/2` conforme ambiente e evitar oversubscription em CPU.
- Avaliar mixed precision (`torch.cuda.amp`) em GPU.
- Persistir checkpoints por evento relevante (melhor acurácia/epoch final), não por batch.

## Recomendação operacional

Para ambiente de CI e validação rápida:
- `train_limit <= 512`
- `epochs=1`
- sem telemetria detalhada
- logging somente em arquivo
