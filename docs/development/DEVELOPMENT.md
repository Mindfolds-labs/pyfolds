# Guia de Desenvolvimento

Guia operacional para desenvolvimento local e contribuição incremental no repositório.

## 1) Pré-requisitos
- Python 3.10+
- `pip` atualizado
- Ambiente virtual recomendado

## 2) Setup local
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 3) Fluxo recomendado
1. Criar branch de trabalho por ISSUE.
2. Implementar mudanças pequenas e rastreáveis.
3. Atualizar documentação afetada na mesma branch.
4. Rodar validações locais antes do commit.

## 4) Validação mínima antes de commit
```bash
python -m compileall src/
PYTHONPATH=src pytest tests/ -v
python tools/check_links.py docs/ README.md
python tools/sync_hub.py --check
```

## 5) Padrões de contribuição
- Preferir alterações pequenas por PR.
- Manter nomes de arquivos consistentes e descritivos.
- Preservar navegação documental (sem links quebrados).
- Registrar mudanças operacionais em `execution_queue.csv` quando aplicável.

## 6) Fechamento da tarefa
- Atualizar artefatos da ISSUE (relatório/log).
- Confirmar status no HUB de controle.
- Garantir que a PR descreve claramente escopo e validações executadas.
