# Nota técnica — impacto da ativação dos mecanismos MPJRD no aprendizado

## Contexto da última implementação

Na implementação mais recente, o FOLDSNet foi consolidado para operar com pilha MPJRD completa (sem backend simplificado), mantendo os mecanismos biológicos ativos no caminho principal de execução.

Objetivo desta nota: verificar, em experimento rápido, se a ativação dos mecanismos MPJRD melhora o aprendizado em relação a uma configuração com os mecanismos desativados.

## Protocolo de comparação

Foram executados dois treinos controlados do modelo `mpjrd` (2 épocas, batch 256, LR 0.001, CPU):

1. **MPJRD ON**: todos os mecanismos ativos.
2. **MPJRD OFF**: mecanismos desativados via `--disable-*`.

Comandos executados:

```bash
python train_mnist_mind.py --model mpjrd --epochs 2 --batch 256 --lr 0.001 --run-id cmp_mpjrd_on --device cpu
python train_mnist_mind.py --model mpjrd --epochs 2 --batch 256 --lr 0.001 --run-id cmp_mpjrd_off --device cpu \
  --disable-stdp --disable-homeostase --disable-inibicao --disable-refratario \
  --disable-backprop --disable-sfa --disable-stp --disable-wave --disable-circadian \
  --disable-engram --disable-speech
```

## Resultado observado

### MPJRD ON (`runs/cmp_mpjrd_on`)
- best_acc: **10.94%**
- final_acc: **10.94%**
- final_loss: **2.3917**

### MPJRD OFF (`runs/cmp_mpjrd_off`)
- best_acc: **10.74%**
- final_acc: **9.18%**
- final_loss: **2.4024**

## Interpretação

Neste teste rápido, a configuração com mecanismos MPJRD ativos apresentou desempenho ligeiramente melhor (maior `best_acc`, maior `final_acc` e menor `final_loss`).

Diferença ainda pequena e com variância esperada para treino curto; para conclusão forte, recomenda-se:

- mais épocas (10–30),
- repetição com múltiplas seeds,
- validação com dataset real fixo e mesma partição em todas as repetições.

## Observação importante

O pipeline pode usar fallback para dados sintéticos se o dataset real não estiver disponível no ambiente. Isso acelera debug, mas reduz validade estatística da comparação.
