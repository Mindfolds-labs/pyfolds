# FOLD/.MIND Risk Register

## Escala usada
- **Probabilidade**: Baixa / Média / Alta
- **Impacto**: Baixo / Médio / Alto

## Riscos

| ID | Risco | Prob. | Impacto | Mitigação | Rollback |
|---|---|---|---|---|---|
| R1 | Divergência entre especificação e implementação real do formato | Média | Alto | Revisão cruzada da spec com testes e leitura do código de serialização | Reverter commit de documentação e reabrir ADR com escopo reduzido |
| R2 | Linguagem normativa inconsistente (MUST/SHOULD/MAY) gerar ambiguidade | Média | Médio | Checklist editorial de termos normativos antes do merge | Reverter docs afetados e publicar errata em novo PR |
| R3 | Testes de corrupção frágeis por dependerem de mensagens exatas de exceção | Média | Médio | Validar por tipo de exceção e palavra-chave mínima quando necessário | Desabilitar teste instável e substituir por versão robusta em PR corretivo |
| R4 | Falso negativo em truncamento por criar arquivo inválido já na etapa base | Baixa | Médio | Garantir fixture com arquivo válido antes de corromper bytes | Reverter testes recém-adicionados e reaplicar com fixture validada |
| R5 | Limites anti-DoS mal calibrados (muito baixos/altos) | Média | Alto | Introduzir valores explícitos na spec e testes de limites | Reverter ajuste de limite e manter valor anterior até recalibração |
| R6 | CI com benchmark lento causar fila e custo excessivos | Alta | Médio | Executar benchmark apenas manual/semanal; evitar em todo push | Reverter workflow de benchmark e manter execução local temporária |
| R7 | Regressão de performance sem detecção por ausência de baseline | Média | Médio | Exportar `benchmark.json` e manter histórico comparável | Reverter mudança de performance e restaurar baseline conhecido |
| R8 | Dependência opcional ausente no runner (pytest-benchmark) | Média | Médio | Instalar dependência no workflow explicitamente por `pip install` | Reverter etapa de benchmark no workflow até ambiente estabilizar |
| R9 | Conflitos entre branches por sobreposição de diretórios | Baixa | Médio | Aplicar política de superfície de alteração por ação | Reverter PR conflitante e reaplicar em branch limpa |
| R10 | Cobertura insuficiente de corrupção (offset/index/chunk) | Média | Alto | Matriz mínima obrigatória de cenários adversariais | Reverter status de pronto e bloquear merge até cobertura mínima |
| R11 | Artefatos de benchmark não publicados por falha de workflow | Média | Médio | Validar upload de artifact em execução manual antes da agenda | Reverter workflow atual e aplicar versão simplificada de artifact-only |
| R12 | Mudança em CI sem documentação de evidência | Baixa | Médio | Exigir checklist de evidência no template de PR | Reverter merge de workflow e reapresentar com evidências completas |

## Observações operacionais
- Revisar este registro a cada ação concluída.
- Qualquer risco com impacto alto sem mitigação validada bloqueia merge.
