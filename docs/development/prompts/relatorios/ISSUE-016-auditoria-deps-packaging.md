# ISSUE-016 — Auditoria de dependências e packaging

## Objetivo
Fechar lacunas de auditoria em empacotamento, compatibilidade de dependências e documentação de depreciação da API pública.

## Diagnóstico (ANALISAR)

### Divergências encontradas
- `pyproject.toml` estava em `2.2.0`, mas `setup.cfg` mantinha `2.1.0` e lista própria de dependências/extras.
- `requirements.txt` misturava runtime com componentes opcionais (ex.: `tensorflow-cpu`, `torchvision`, `safetensors`), divergindo de `[project.dependencies]`.
- Havia configuração duplicada de pytest entre `pyproject.toml` e `pytest.ini`.
- Documentação de depreciação (`README.md`, `CHANGELOG.md`, `PUBLIC_API_V2_CONTRACT.md`) mencionava remoção dos aliases v1 em `2.0.0`, mas o comportamento real em `src/pyfolds/__init__.py` ainda mantém aliases com `DeprecationWarning` na versão `2.2.0`.

## Execução (EXECUTAR)
- `pyproject.toml` mantido como fonte canônica e recebeu marcador de ambiente para `tensorflow-cpu` nos extras (`python_version >= '3.9'`).
- `setup.cfg` simplificado para evitar drift de metadados/deps.
- `requirements.txt` restringido ao conjunto runtime canônico.
- Contrato documental de depreciação v1 atualizado para janela `2.x` e remoção planejada em `3.0.0`.
- Configuração de pytest consolidada em `pytest.ini` (remoção do bloco duplicado no `pyproject.toml`).

## Matriz de instalação (dry-run)

| Python alvo | Método | Resultado |
| --- | --- | --- |
| 3.8 | `pip --python-version 3.8` (simulação resolver) | OK para instalação base (`.`) |
| 3.9 | `pip --python-version 3.9` (simulação resolver) | OK para instalação base (`.`) |
| 3.10 | `PYENV_VERSION=3.10.19 python -m pip install --dry-run -e .` | OK |
| 3.11 | `PYENV_VERSION=3.11.14 python -m pip install --dry-run -e .` | OK |

## Riscos mitigados
- Eliminado drift entre manifestos de packaging.
- Explicitado comportamento de extras com TensorFlow por versão de Python.
- Removida fonte comum de warning/config drift no pytest.
- Documentação alinhada ao contrato real da API pública.

## Pendências
- Validação completa de extras (`.[dev]`) em Python 3.8/3.9 depende de execução nativa com interpretadores dessas versões (não apenas simulação do resolver com `--python-version`).
