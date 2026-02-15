# METHODOLOGY — Hipótese, desenho experimental e critérios de falsificação

## 1) Hipótese científica

> **H1:** Computação dendrítica local (com compartimentalização por ramo) provê separabilidade funcional suficiente para reduzir ou dispensar camadas ocultas externas em tarefas selecionadas.

Hipótese nula:

> **H0:** Sem camadas ocultas externas, o modelo não supera baseline linear em regime equivalente de parâmetros.

## 2) Racional teórico

A premissa é que cada dendrito funciona como um subcomputador não linear, gerando features locais antes da agregação somática. Esse arranjo introduz profundidade funcional interna ao neurônio, mesmo em topologias sem múltiplas camadas explícitas.

## 3) Variáveis experimentais

- **Independentes:**
  - número de dendritos (`D`), sinapses por dendrito (`S`), thresholds, modo de neuromodulação.
  - presença/ausência de competição WTA.
  - modo de treino (`ONLINE`, `BATCH`, `SLEEP`, `INFERENCE`).
- **Dependentes:**
  - acurácia/F1, robustez a ruído, taxa de disparo, saturação sináptica.
- **Controle:**
  - seed fixa, partição de dados constante, mesmo orçamento de parâmetros para comparações.

## 4) Desenho de estudos

### Estudo A — Separabilidade
Comparar MPJRD (compartimentalizado) vs. perceptron linear com orçamento similar.

### Estudo B — Ablação estrutural
Remover não linearidade local dendrítica ou WTA e medir queda de desempenho.

### Estudo C — Continual learning
Aplicar treinos sequenciais com ciclos de consolidação (`sleep`) e medir retenção.

## 5) Critérios de evidência

Aceitar H1 quando, de forma reprodutível:
- MPJRD superar baseline linear em tarefa não linear.
- Ablation sem compartimentalização reduzir desempenho de forma consistente.
- Métricas internas (`θ`, `r_hat`, saturação, proteção) permanecerem estáveis.

## 6) Ameaças à validade

- Confundir ganho de performance com aumento de parâmetros.
- Overfitting por tuning excessivo.
- Métricas internas sem correlação causal com performance externa.

## 7) Reprodutibilidade

- Salvar configuração (`MPJRDConfig`) por experimento.
- Registrar seed e versão do commit.
- Exportar métricas de telemetria por etapa.
