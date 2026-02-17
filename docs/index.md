# PyFolds Documentation

Documentação técnica estruturada para uso, desenvolvimento, API e governança do PyFolds.

## Objetivo
Estabelecer uma base documental estável, navegável e consistente com padrões técnicos IEEE-like.

## Escopo
- Guias de usuário (instalação, quickstart, exemplos).
- Guias de desenvolvimento (contribuição, testes, empacotamento, diagramas).
- Estratégia de documentação de API.
- Governança (ADRs e políticas de qualidade).
- **Hub interno existente:** `docs/development/HUB_CONTROLE.md` (não recriar do zero; apenas evoluir).

## Definições/Termos
- **IEEE-like:** estilo técnico com seções formais, rastreabilidade e linguagem objetiva.
- **ADR:** *Architecture Decision Record*.
- **MyST:** sintaxe Markdown estendida para Sphinx.

## Conteúdo técnico

```{raw} html
<div class="pyfolds-hero">
  <img src="_static/brand/pyfolds-logo.svg" alt="PyFolds brand mark" class="pyfolds-hero-logo" />
  <p class="pyfolds-hero-subtitle">Framework neural com identidade visual técnica, dinâmica e orientada a engenharia.</p>
</div>
```

```{toctree}
:maxdepth: 2
:caption: Navegação principal

user/index
development/index
api/index
governance/index
```

## Referências
- [README do projeto](../README.md)
- [Portal interno de prompts](development/prompts.md)
