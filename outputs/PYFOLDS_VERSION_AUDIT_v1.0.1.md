# Relatório de Auditoria de Versão e Instalação — PyFolds v1.0.1

## Objetivo
Validar arquivos de instalação/empacotamento, consistência de versão da API (`__init__`) e integridade básica de imports para permitir recompilação/republicação com `pip install`.

## Escopo auditado
- Empacotamento:
  - `pyproject.toml`
  - `setup.cfg`
  - `setup.py`
- Metadados de versão da API:
  - `src/pyfolds/__init__.py`
  - `src/pyfolds/core/__init__.py`
  - `src/pyfolds/network/__init__.py`
  - `src/pyfolds/layers/__init__.py`
- Verificações de runtime:
  - compilação dos módulos (`compileall`)
  - smoke test de importações
  - instalação editable via pip (`pip install -e .`)

## O que foi encontrado
1. A versão publicada no projeto estava em `2.0.0` nos arquivos principais de distribuição e API.
2. Os módulos principais expunham `__version__` com `2.0.0`.
3. A estrutura de imports dos pacotes principais estava funcional após validação com `PYTHONPATH=src`.

## O que foi corrigido
1. **Versão do pacote alterada para `1.0.1`** nos pontos canônicos de distribuição:
   - `pyproject.toml` (`[project].version`)
   - `setup.cfg` (`[metadata].version`)
2. **Versão da API alinhada para `1.0.1`**:
   - `src/pyfolds/__init__.py`
   - `src/pyfolds/core/__init__.py`
   - `src/pyfolds/network/__init__.py`
   - `src/pyfolds/layers/__init__.py`
3. **Dependências revisadas**:
   - requisito de `torch` mantido corretamente em `torch>=2.0.0` (não foi rebaixado com a mudança de versão do pacote).

## Verificações executadas
1. `python -m compileall src/pyfolds`
   - Resultado: OK (compilação concluída sem erro).
2. `PYTHONPATH=src python - <<'PY' ... importlib.import_module(...) ... PY`
   - Resultado: OK.
   - Versões reportadas:
     - `pyfolds 1.0.1`
     - `core 1.0.1`
     - `network 1.0.1`
     - `layers 1.0.1`
3. `python -m pip install -e .`
   - Resultado: OK.
   - Wheel editable gerada e instalada como `pyfolds-1.0.1`.

## Status final
- ✅ Versão consolidada para **v1.0.1** nos arquivos principais de distribuição e API.
- ✅ Imports principais do pacote carregam corretamente.
- ✅ Instalação com pip (editable) concluída com sucesso.
- ✅ Projeto pronto para recompilação/reempacotamento com a versão solicitada.
