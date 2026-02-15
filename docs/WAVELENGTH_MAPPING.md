# WAVELENGTH_MAPPING — Estratégia de frequências por classe

## Objetivo
Mapear 10 classes (ex.: MNIST) para 10 frequências fundamentais com separação suficiente para leitura robusta em tempo discreto.

## Proposta base
Para `c ∈ {0..9}`:

`f_c = f_base + c * Δf`

Sugestão inicial:
- `f_base = 12 Hz`
- `Δf = 4 Hz`
- faixa total: 12–48 Hz

## Critérios de separação
1. **Resolução temporal do simulador**
   - Com passo `dt` (s), frequência de Nyquist: `f_N = 1/(2dt)`.
   - Exemplo `dt = 1 ms` ⇒ `f_N = 500 Hz` (faixa 12–48 segura).
2. **Janela de observação T**
   - Resolução espectral aproximada: `δf ≈ 1/T`.
   - Para distinguir classes com `Δf = 4 Hz`, usar `T >= 250 ms` para leitura espectral fina.
3. **Evitar interferência destrutiva sistêmica**
   - Não usar frequências muito próximas a harmônicos dominantes da dinâmica interna.
   - Usar jitter pequeno opcional por classe em treino para robustez.

## Estratégias alternativas
- **Banda gamma compacta:** 30–75 Hz (Δf=5 Hz)
- **Escala logarítmica:** `f_c = f_base * r^c` para separar proporcionalmente em bandas largas.

## Recomendação para início em MNIST
- Início conservador: 12–48 Hz, `dt=1 ms`.
- Medir confusão interclasse por coerência de fase e energia por canal.
- Ajustar `Δf` até minimizar sobreposição espectral na validação.
