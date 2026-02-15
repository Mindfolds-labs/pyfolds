# Fundamentos Científicos do PyFolds

## 1. Introdução

O PyFolds implementa um modelo de neurônio artificial baseado em descobertas recentes da neurociência computacional. Diferente de redes neurais tradicionais, o PyFolds expõe explicitamente mecanismos biofísicos de processamento dendrítico, plasticidade sináptica e consolidação de memória.

## 2. Pilares Científicos

### 2.1 Quantização Sináptica (Bartol et al., 2015)
- Sinapses reais apresentam estados discretos funcionais.
- Implementação: contador `N` por sinapse (0-31).
- Peso: `W = log2(1+N) / w_scale`.

### 2.2 Subunidades Dendríticas (Poirazi & Mel, 2001)
- Dendritos como unidades computacionais independentes.
- Não-linearidade local antes da soma somática.

### 2.3 Spikes Dendríticos (Gidon et al., 2020)
- Possibilidade de spikes locais regenerativos.
- Limiar e ganho por dendrito para coincidência e seletividade.

### 2.4 Codificação por Fase
- Informação representada também no tempo relativo do spike.
- v3.0: fase + amplitude para enriquecer representação.

## 3. Modelo Matemático Formal

### 3.1 Sinapse
`N ∈ [0, 31] ⊂ ℕ`  
`I ∈ ℝ`  
`W = log2(1 + N) / w_scale`

### 3.2 Dendrito
`v_d = Σ_s W_s * x_s`  
`a_d = σ(γ * (v_d - θ_d))`

### 3.3 Soma
`u = Σ_d a_d`  
`spike = 1 se u ≥ θ, 0 caso contrário`

### 3.4 Plasticidade Three-Factor
`ΔI = η * R * Hebb * (1 + βW) * dt`

### 3.5 Consolidação (Sono)
`ΔN = round(eligibility * consolidation_rate)`

## 4. Validação Experimental
- XOR com um único neurônio dendrítico.
- MNIST sem camadas ocultas, com especialização funcional.
- Robustez a ruído e variações de entrada.

## 5. Implicações para IA
1. Interpretabilidade.
2. Eficiência amostral.
3. Consolidação offline.
4. Processamento temporal por fase.
