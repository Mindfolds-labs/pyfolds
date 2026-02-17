# Diagrama de Sequência

## Objetivo
Modelar a interação entre contribuidor e pipeline de documentação.

## Escopo
Da edição local até publicação em páginas de documentação.

## Definições/Termos
- **Pipeline:** automação de validação e build.

## Conteúdo técnico
Figura 1 — Sequência de contribuição.

```{plantuml}
@startuml
actor Contribuidor as C
participant "Repositório" as R
participant "Validador" as V
participant "Builder Sphinx" as B
participant "GitHub Pages" as G

C -> R : commit de docs
R -> V : check_links / sync_hub
V --> R : status
R -> B : build html
B --> R : artefatos
R -> G : deploy
G --> C : documentação publicada
@enduml
```

Legenda: mensagens descrevem eventos de integração contínua.

## Referências
- [Style Guide](STYLE_GUIDE.md)
