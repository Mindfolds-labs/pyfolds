# Observabilidade com TensorBoard

Use `TensorBoardLogger` para métricas, histogramas, embeddings e sinais de engrams.

```python
from pyfolds.visualization import TensorBoardLogger
logger = TensorBoardLogger("./runs/noetic")
logger.log_scalar("train/loss", 0.1, step=1)
```
