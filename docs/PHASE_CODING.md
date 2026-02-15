# PHASE_CODING — Matemática operacional MPJRD-Wave

## 1) Integração dendrítica cooperativa
Para entrada `x`:
- Linear local: `v_d = Σ_s W_{d,s} x_{d,s}`
- Não-linear local: `a_d = sigmoid(v_d - τ_d)`
- Soma cooperativa: `U = Σ_d a_d`

## 2) Disparo
`spike = 1[U >= θ]`, com `θ` adaptado por homeostase.

## 3) Amplitude (escala estrutural)
`A = log2(1 + U)`

Mantém coerência com a parametrização estrutural baseada em filamentos/sinapses discretas.

## 4) Fase a partir da latência/certeza
Implementação escolhida:

`ϕ = (π/2) * (1 - sigmoid((U - θ) * k_ϕ))`

- Se `U` sobe acima de `θ`, `ϕ` cai (resposta mais adiantada).
- `k_ϕ` controla sensibilidade da fase.

Latência auxiliar usada para análise:

`lat = λ / (A + ε)`

## 5) Onda de saída
Para frequência de categoria `f_c`:

`S(t) = A * cos(2π f_c t + ϕ)`

Saída em quadratura:
- `real = A * cos(2π f_c t + ϕ)`
- `imag = A * sin(2π f_c t + ϕ)`

## 6) Sincronia de fase na plasticidade
Define-se sincronia local:

`sync = cos(ϕ_atual - ϕ_ref)`

Sinal neuromodulador efetivo:

`R_eff = clip(R * (1 + g_sync * sync), -1, 1)`

Com isso, conexões que disparam em fase com o contexto/recompensa recebem maior reforço.

## 7) Relação com MNIST
- **Frequência (`f_c`)**: canal de classe (0–9).
- **Fase (`ϕ`)**: prioridade temporal/certeza.
- **Amplitude (`A`)**: intensidade agregada do padrão.
