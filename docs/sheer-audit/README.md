# Sheer Audit integration workspace

Esta pasta centraliza o material de preparação para integrar o **Sheer Audit** na documentação do projeto.

## O que foi feito

1. Repositório de referência clonado localmente em `/workspace/sheer-audit`.
2. Instalação em modo editável com `pip install -e .`.
3. Testes do Sheer Audit executados com sucesso (ver `reports/sheer_audit_pytest.txt`).
4. Geração de inventário de arquivos Python do `pyfolds` com os módulos públicos atuais do Sheer (`config` + `scan/repo.py`).

## Conteúdo desta pasta

- `sheer.toml`: configuração inicial para uso futuro no `pyfolds`.
- `reports/`: logs e relatórios operacionais.
- `data/`: saídas JSON do inventário de arquivos Python por perfil.
- `data/code_map.md` e `data/code_map.json`: mapa do código do `pyfolds`.
- `data/repo_model.json`: modelo de repositório no schema Sheer (`RepoModel`).
- `data/uml/*.puml`: UML de pacotes e classes para renderização futura.
- `blueprints/integracao-futura.md`: guia/blueprint de integração gradual.
- `blueprints/bluesprinta-mapa-uml.md`: blueprint específico para mapa + UML.
- `scripts/generate_repo_model.py`: gerador AST -> `RepoModel` para baseline.

## Observação importante

A CLI atual disponível no pacote instalado é um bootstrap e ainda não expõe comandos funcionais (`scan/report/uml/trace`) neste ambiente.
Por isso, os dados foram extraídos com os módulos Python existentes e estáveis do projeto (`load_config`, `collect_python_files`).
