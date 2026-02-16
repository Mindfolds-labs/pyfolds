# Análise de Bugs

## Escopo

Documento de triagem inicial para classificar falhas recorrentes e orientar priorização técnica.

## Critérios de classificação

- **Severidade:** crítica, alta, média, baixa.
- **Impacto funcional:** bloqueante, degradante, cosmético.
- **Risco sistêmico:** local, módulo, transversal.

## Mapa resumido

| Categoria | Severidade típica | Área principal | Observação |
|---|---|---|---|
| Serialização/checkpoint | Alta | `src/pyfolds/serialization` | Pode afetar reprodutibilidade e retomada de treino |
| Telemetria/logging | Média | `src/pyfolds/telemetry` e `src/pyfolds/utils/logging.py` | Impacta observabilidade e diagnóstico |
| Mecanismos avançados | Média/Alta | `src/pyfolds/advanced` | Exige testes de integração consistentes |
| Camada de rede/wave | Média | `src/pyfolds/network` e `src/pyfolds/wave` | Falhas aparecem sob carga e cenários extremos |

## Fluxo recomendado

1. Registrar bug com cenário mínimo reproduzível.
2. Associar teste automatizado de regressão.
3. Definir correção incremental e validação em CI.
4. Atualizar documentação afetada.

