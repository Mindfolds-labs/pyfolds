---
id: "ISSUE-042"
titulo: "Análise das sugestões Gemini para PyFolds com validação matemática"
prioridade: "Alta"
area: "Core/LEIBREG/Noetic"
fase: "legado"
adr_vinculada: "ADR 0041"
normas:
  - ABNT NBR ISO/IEC 25010
  - IEEE 830
---

# ISSUE-042: Análise das sugestões Gemini para PyFolds com validação matemática

## Metadados
- **Fase:** `legado`
- **ADR vinculada:** `ADR 0041`

## Objetivo
Mapear, no código atual do PyFolds, quais sugestões do relatório Gemini são aplicáveis de forma objetiva e segura, distinguindo itens já cobertos, lacunas reais e riscos matemáticos de implementação.

## Contexto Técnico
A análise externa cita módulos que **não existem** neste repositório (ex.: `align.py`, `rive_mpjrd.py`, `associative_memory.py`), então foi necessário fazer correspondência por função matemática equivalente nas camadas atuais (`core`, `leibreg`, `noetic_pawp`).

## Análise Técnica

### 1) Paralelização dos dendritos — **Alta prioridade / aplicável já**
- Evidência atual: o `forward` de `MPJRDNeuron` ainda usa loop Python por dendrito e posterior `stack`.
- Risco matemático: nenhum, desde que a forma de agregação por dendrito seja preservada.
- Ganho esperado: redução de overhead de Python no hot path.
- Recomendação segura:
  1. Introduzir caminho vetorizado opcional (`dendrite_vectorized=True`) para equivalência A/B.
  2. Validar invariantes: `v_dend`, `u`, `spikes`, estatísticas e gradientes.
  3. Liberar por feature-flag após benchmark.

### 2) Legendre na GPU — **Alta prioridade / não aplicável diretamente no estado atual**
- Não foi encontrado no PyFolds atual um projetor Legendre no `forward` que use `np.linalg.lstsq`.
- Conclusão: a sugestão é tecnicamente boa, mas parece referir outro código-base.
- Recomendação segura: abrir tarefa somente quando o módulo de projeção espectral for introduzido neste repositório.

### 3) DTW para alinhamento fonético — **Alta prioridade conceitual / fora do escopo atual de PyFolds**
- Não há pipeline de alinhamento grafema↔fonema no código atual.
- Conclusão: sugestão correta para stack de ASR/TTS, porém não mapeada para os módulos presentes.

### 4) Uncertainty weighting de perdas — **Média prioridade / aplicável no treino multitarefa**
- O mecanismo é matematicamente consistente com Kendall et al. (2018):
  \[
  \mathcal{L}_{total}=\sum_i \frac{1}{2\sigma_i^2}\mathcal{L}_i + \frac{1}{2}\log\sigma_i^2
  \]
- Recomenda-se implementação em módulo separado de treino, com clamp de `log_vars` e telemetria para evitar colapso numérico.

### 5) LSH para memória associativa — **Média prioridade / parcial no design atual**
- LEIBREG atual usa `EngramBank` por ressonância/cosseno, sem LSH explícito.
- Recomendação segura: camada opcional de indexação aproximada para recall em grande escala, sem alterar semântica de ranking final.

### 6) Continuidade C1/C2 em Bézier — **Baixa prioridade / sem alvo direto no estado atual**
- Não há módulo explícito de reconstrução por segmentos Bézier no núcleo atual.
- Conclusão: adiar até existir pipeline geométrico correspondente.

### 7) DEQ no REGCore — **Baixa prioridade / alto risco de estabilidade**
- `REGCore` atual é residual empilhado com normalização, de comportamento previsível.
- DEQ exigiria solver de ponto fixo + backward implícito; risco de não convergência sem restrições de Lipschitz.
- Recomendação: manter apenas como trilha de pesquisa, fora do caminho principal.

## Requisitos Funcionais
- [ ] RF-01 Criar PR incremental para vetorização dos dendritos com equivalência numérica comprovada.
- [ ] RF-02 Criar PR incremental para perda multitarefa por incerteza com logs de estabilidade.

## Requisitos Não-Funcionais
- [ ] RNF-01 Performance: reduzir latência do `forward` no modo padrão.
- [ ] RNF-02 Confiabilidade: garantir saída finita e convergência de treino.
- [ ] RNF-03 Reprodutibilidade: testes determinísticos e benchmark com semente fixa.

## Critérios de Aceite
- [ ] Nenhuma alteração semântica no comportamento padrão quando flags novas estiverem desativadas.
- [ ] Testes de regressão numérica aprovados para `v_dend`, `u`, `spikes`.
- [ ] Métricas de tempo e memória documentadas antes/depois.

## Riscos e Mitigações
- **Risco:** Vetorização alterar discretamente o acúmulo numérico em float.
  - **Mitigação:** comparar com tolerância explícita (`atol`, `rtol`) e seed fixa.
- **Risco:** Uncertainty weighting favorecer tarefa trivial.
  - **Mitigação:** monitorar `log_vars` e impor faixa operacional.
- **Risco:** DEQ instável.
  - **Mitigação:** manter fora do release path.

## Referências científicas
1. Kendall, A.; Gal, Y.; Cipolla, R. *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*. CVPR, 2018.
2. Cuturi, M.; Blondel, M. *Soft-DTW: a Differentiable Loss Function for Time-Series*. ICML, 2017.
3. Bai, S.; Kolter, J. Z.; Koltun, V. *Deep Equilibrium Models*. NeurIPS, 2019.
