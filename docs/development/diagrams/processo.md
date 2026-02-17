# Diagrama de Processo

## Objetivo
Demonstrar fluxo operacional de build de documentação.

## Escopo
Pipeline local de validação e publicação.

## Definições/Termos
- **Build docs:** geração de HTML com Sphinx.

## Conteúdo técnico
Figura 1 — Fluxo de documentação.

```{mermaid}
flowchart TD
  A[Editar conteúdo] --> B[Validar links]
  B --> C[Build Sphinx HTML]
  C --> D{Sucesso?}
  D -- Sim --> E[Publicar artefatos]
  D -- Não --> F[Corrigir inconsistências]
  F --> B
```

Legenda: caixas representam etapas; losango representa decisão.

## Referências
- [Style Guide](STYLE_GUIDE.md)
