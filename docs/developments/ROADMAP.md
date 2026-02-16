# Roadmap

## Objetivo
Garantir que o PyFolds evolua com foco em **segurança de serialização**, **estabilidade matemática** e **execução confiável em produção**.

## Sprint 0 — Hardening de serialização e mecanismos avançados (concluído)

### Entregas concluídas
- Correção de fechamento seguro de recursos no `FoldReader.__exit__`.
- Escrita robusta em `finalize()` com flush + `os.fsync()` para reduzir risco de inconsistência.
- Validações defensivas de leitura em `_read_at()` para evitar leituras fora do limite.
- Fortalecimento de `_read_header_and_index()` com validações estruturais e limite de tamanho do índice.
- Melhoria de observabilidade em `advanced/__init__.py` com logs de inicialização dos mixins.
- Clareza semântica em STDP (`[B, D, S]`) para reduzir ambiguidades de shape e broadcast.

### Critérios de aceite atendidos
- Zero regressões nos testes de serialização.
- Zero regressões nos testes de mecanismos avançados.
- Sem vulnerabilidades críticas conhecidas no parser `.fold/.mind`.

---

## Sprint 1 — Estabilidade numérica e invariantes (prioridade alta)

### Backlog
1. Implementar *safe weight law* (clamp + validação `NaN/Inf`) no cálculo de pesos.
2. Adicionar monitor de saúde com checagens periódicas de invariantes (`N`, `I`, `theta`).
3. Validar limites de homeostase com estratégia de controle mais estável (anti-windup).
4. Incluir testes baseados em propriedades para limites e monotonicidade de atualização.

### DoD (Definition of Done)
- Testes novos aprovados (`unit` + propriedades).
- Sem `NaN/Inf` em execução prolongada de teste sintético.
- Métricas de saúde expostas em API de monitoramento.

---

## Sprint 2 — Robustez de formato e recuperação (prioridade alta)

### Backlog
1. Adicionar checksum hierárquico opcional por chunk + manifesto.
2. Implementar testes de corrupção: truncamento, offset inválido, checksum inválido.
3. Criar modo de leitura com degradação segura (erro explícito + diagnóstico).
4. Documentar política de versionamento de formato `.fold/.mind`.

### DoD
- Cobertura dos cenários de corrupção críticos.
- Mensagens de erro acionáveis para operação.
- Documento de compatibilidade de versões publicado.

---

## Sprint 3 — Performance e escala (prioridade média)

### Backlog
1. Benchmark de serialização e desserialização por tamanho de modelo.
2. Benchmark dos mecanismos avançados com lotes grandes.
3. Loader com estratégia de memória eficiente para checkpoints grandes.
4. Relatório de trade-off: ECC/compressão vs latência/throughput.

### DoD
- Relatório com baseline e metas de performance.
- Piores casos mapeados e com mitigação definida.

---

## Sprint 4 — Produção e operação (prioridade média)

### Backlog
1. Runbook operacional para falhas de leitura/gravação.
2. Alertas e telemetria para integridade de checkpoint.
3. Checklist formal pré-release para serialização e treinamento.

### DoD
- Runbook revisado.
- Checklist de release integrado ao processo.
- Pipeline com validações automatizadas de qualidade.
