# Issue 005 — Confiabilidade de infraestrutura: reconfiguração de logging e retenção de telemetria

**Tipo:** bug de infraestrutura (estado global e observabilidade)  
**Severidade:** média-alta  
**Status:** corrigido

## Problema A — Logging não reconfigurável e sem singleton real

A classe `PyFoldsLogger` instanciava novo objeto a cada chamada de construtor e executava `setup()` automático com guarda de inicialização que bloqueava reconfigurações subsequentes.

Efeitos observáveis:
- Quebra da expectativa de singleton (`PyFoldsLogger() is PyFoldsLogger()` falso).
- `setup(level="TRACE")` podia não surtir efeito após setup inicial em `INFO`.
- `setup(log_file=...)` posterior podia não criar handler de arquivo.
- Níveis por módulo podiam não ser aplicados quando setup anterior já havia ocorrido.

## Problema B — Capacidade padrão de telemetria truncava amostras em testes de amostragem

`TelemetryConfig.memory_capacity` padrão de `512` levava a truncamento em cenário de amostragem probabilística (~500-600 eventos emitidos em 1000 chamadas), gerando inconsistência entre "eventos emitidos" e "eventos retidos" em testes.

## Correções aplicadas

1. **Singleton robusto para `PyFoldsLogger`** via `__new__` + guarda de bootstrap em `__init__`.
2. **`setup()` reconfigurável**: sempre reaplica configuração, remove/fecha handlers antigos e reinstala handlers conforme parâmetros atuais.
3. **Capacidade padrão da telemetria aumentada para 2048**, evitando truncamento em cenários usuais de validação de amostragem sem alterar semântica de emissão.

## Impacto técnico

- Estado global de logging fica determinístico e reproduzível entre módulos e testes.
- Telemetria mantém fidelidade de observação em janelas curtas de benchmark/teste.
- Sem alteração no modelo matemático do neurônio; mudança restrita a infraestrutura e mensuração.

## Validação

- Subconjunto solicitado (core + telemetry + utils + integração avançada) passou integralmente.
- Suite completa do repositório passou (136 testes), com warning conhecido de marker `performance` não registrado.
