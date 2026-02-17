# Bluesprinta: Mapa de Código + UML do pyfolds

## Entregáveis gerados agora

- Mapa estruturado em JSON: `docs/sheer-audit/data/code_map.json`
- Mapa legível em Markdown: `docs/sheer-audit/data/code_map.md`
- UML de pacotes (PlantUML): `docs/sheer-audit/data/uml/package.puml`
- UML de classes (visão geral): `docs/sheer-audit/data/uml/class_overview.puml`
- Modelo Sheer (schema): `docs/sheer-audit/data/repo_model.json`

## Como manter atualizado

Sempre que houver alteração relevante em `src/pyfolds`:

1. Regenerar o mapa + UML.
2. Revisar mudanças em `code_map.md`.
3. Confirmar se as dependências entre pacotes continuam coerentes.
4. Commitar artefatos junto da mudança de arquitetura.

## Uso futuro na documentação

- Publicar seção "Mapa do Código" usando `code_map.md`.
- Renderizar `*.puml` no pipeline de docs para gerar PNG/SVG.
- Conectar esses artefatos ao fluxo de auditoria Sheer.
