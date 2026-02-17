# Diagrama de Componentes

## Objetivo
Representar macrocomponentes da documentação PyFolds.

## Escopo
Relacionamentos entre User/Development/API/Governance e fontes de ativos.

## Definições/Termos
- **Componente:** unidade funcional de documentação.

## Conteúdo técnico
Figura 1 — Componentes documentais.

```{plantuml}
@startuml
skinparam componentStyle rectangle
skinparam defaultTextAlignment center

component "User Docs" as USER
component "Development Docs" as DEV
component "API Docs" as API
component "Governance Docs" as GOV
database "Static Assets" as ASSETS

USER --> API : consulta referências
DEV --> API : valida cobertura
DEV --> GOV : aplica políticas
USER --> ASSETS : consome mídia
DEV --> ASSETS : publica diagramas
@enduml
```

Legenda: setas sólidas indicam dependência direta.

## Referências
- [Style Guide](STYLE_GUIDE.md)
