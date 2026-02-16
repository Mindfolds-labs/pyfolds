# Prompts de Melhoria para o PyFolds

Este documento reúne prompts prontos para usar com assistentes de código (ChatGPT/Codex/Copilot)
com foco nas melhorias mais relevantes para o projeto.

## Como usar

- Copie um prompt por vez.
- Execute em branch dedicada.
- Sempre peça para o assistente: **gerar testes + código + atualização de docs**.
- Ao final, valide com `pytest` e registre no `CHANGELOG.md`.

---

## 1) Fortalecer cobertura de testes por mecanismo

```text
Você está trabalhando no projeto PyFolds.
Objetivo: aumentar cobertura de testes unitários para os mecanismos centrais.

Tarefas:
1. Mapear os módulos em src/pyfolds/core (accumulator, homeostasis, neuromodulation, synapse e neuron_v2).
2. Criar ou expandir testes em tests/unit/core cobrindo:
   - casos nominais;
   - limites numéricos;
   - entradas inválidas;
   - estabilidade temporal (vários steps).
3. Garantir nomes descritivos de teste e uso de fixtures reutilizáveis.
4. Incluir assertions de comportamento (não apenas de tipo/shape).
5. Se detectar bug, corrigir no código de produção e adicionar teste de regressão.

Critérios de aceite:
- Todos os testes novos passam com pytest.
- Cobertura dos módulos core aumenta de forma mensurável.
- Nenhum breaking change na API pública.
```

## 2) Testes de integração entre mixins avançados

```text
No PyFolds, implemente testes de integração para combinação de mixins em src/pyfolds/advanced.

Tarefas:
1. Criar cenários combinando pelo menos 2 mecanismos por teste (ex.: STDP + refractory, adaptation + inhibition).
2. Validar ordem de atualização temporal e efeitos acumulados no estado interno.
3. Cobrir interações com parâmetros extremos para evitar instabilidades.
4. Garantir determinismo usando seeds fixas quando houver aleatoriedade.
5. Documentar em comentários curtos o racional biológico de cada teste.

Critérios de aceite:
- Testes em tests/unit/advanced passam localmente.
- Há pelo menos 1 teste de integração para cada mixin principal.
- Sem duplicação excessiva: usar helpers/fixtures.
```

## 3) Robustez da serialização .fold (corrupção + ECC)

```text
Atue no PyFolds para fortalecer testes de serialização .fold em src/pyfolds/serialization.

Tarefas:
1. Adicionar testes que simulem corrupção de bytes, truncamento de arquivo e checksum inválido.
2. Verificar quando a recuperação por ECC deve funcionar e quando deve falhar explicitamente.
3. Garantir mensagens de erro claras e tipadas.
4. Testar leitura parcial/streaming em chunks.
5. Incluir testes de compatibilidade básica entre versões de checkpoint quando aplicável.

Critérios de aceite:
- Testes em tests/unit/serialization cobrem casos de corrupção realistas.
- Fluxos de falha retornam exceções previsíveis.
- Nenhuma regressão nos testes existentes.
```

## 4) Benchmarks de performance e memória

```text
No projeto PyFolds, crie uma suíte mínima de benchmarks reprodutíveis.

Tarefas:
1. Definir cenários: neurônio único, mini-rede, rede com mixins ativados.
2. Medir latência por step, throughput e uso de memória.
3. Organizar em tests/perf ou benchmarks/ com execução opcional.
4. Registrar baseline inicial em markdown.
5. Evitar flakiness: warm-up e número fixo de iterações.

Critérios de aceite:
- Benchmarks executam sem quebrar o fluxo de testes unitários.
- Saída inclui métricas comparáveis entre runs.
- Documento com baseline e instruções de execução.
```

## 5) Exemplo completo e didático (quickstart)

```text
Melhore a experiência de onboarding do PyFolds criando um exemplo completo e didático.

Tarefas:
1. Criar exemplo em examples/ que inclua: criação de neurônio/rede, passo temporal e leitura de métricas.
2. Adicionar comentários curtos explicando cada bloco.
3. Incluir equivalente em docs (passo a passo para iniciantes).
4. Garantir que o exemplo seja executável em CPU sem dependências extras.
5. Validar que o código do exemplo segue estilo do projeto.

Critérios de aceite:
- Exemplo roda do início ao fim.
- Documentação permite reprodução por alguém novo no projeto.
- Links adicionados no README.
```

## 6) Telemetria prática com múltiplos sinks

```text
No PyFolds, criar um exemplo avançado de telemetria usando src/pyfolds/telemetry.

Tarefas:
1. Demonstrar configuração de perfis (off/light/heavy).
2. Encadear sinks de memória + console + arquivo.
3. Exibir uso de lazy evaluation para eventos custosos.
4. Mostrar como evitar overhead excessivo em produção.
5. Incluir teste simples que valide emissão de eventos críticos.

Critérios de aceite:
- Exemplo funcional em examples/.
- Teste automatizado cobrindo ao menos um fluxo de emissão.
- Documentação curta explicando trade-offs.
```

## 7) Prompt para hardening de API pública

```text
Faça uma revisão de estabilidade de API no PyFolds.

Tarefas:
1. Inventariar símbolos exportados em src/pyfolds/__init__.py e módulos principais.
2. Identificar inconsistências de naming, defaults e tipos.
3. Propor melhorias backward-compatible.
4. Adicionar testes de contrato para API pública crítica.
5. Atualizar docs de referência quando houver ajuste.

Critérios de aceite:
- Sem quebras de compatibilidade em release minor.
- Testes de contrato cobrindo imports e assinaturas esperadas.
- Changelog atualizado com alterações de API.
```

## 8) Prompt para roadmap técnico (3 sprints)

```text
Monte um roadmap técnico de 3 sprints para o PyFolds com foco em qualidade e adoção.

Entregáveis:
1. Sprint 1 (qualidade): testes, cobertura e correções críticas.
2. Sprint 2 (produto): exemplos, docs de uso e ergonomia de API.
3. Sprint 3 (escala): benchmarks, otimizações e observabilidade.

Para cada sprint, detalhe:
- objetivos;
- tarefas;
- riscos;
- métricas de sucesso;
- definição de pronto.

Restrições:
- Priorizar impacto alto e baixo risco.
- Não incluir refactors grandes sem cobertura de testes.
- Entregar formato em tabela markdown.
```

---

## Prompt mestre (copiar e usar quando quiser “pacote completo”)

```text
Você é mantenedor sênior do projeto PyFolds.
Quero um pacote completo de melhoria incremental e seguro.

Faça nesta ordem:
1) Diagnóstico rápido de lacunas em testes, docs e exemplos.
2) Plano de implementação em pequenas PRs (com escopo de 1 dia cada).
3) Implementação da PR #1 com testes e documentação.
4) Relatório final com:
   - arquivos alterados;
   - decisões técnicas;
   - riscos;
   - próximos passos.

Regras:
- Não quebrar API pública sem justificativa explícita.
- Sempre incluir teste de regressão ao corrigir bug.
- Preferir mudanças pequenas, revisáveis e com alto valor.
- Seguir estilo e padrões já usados no repositório.
```
