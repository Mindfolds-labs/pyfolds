# Prompt pronto para próxima execução (copiar/colar no Codex)

Você é um engenheiro sênior responsável por **fechar as lacunas da auditoria técnica do PyFolds** com execução rastreável no fluxo `CRIAR → ANALISAR → EXECUTAR → FINALIZAR`.

## Objetivo
Implementar correções de governança de empacotamento, compatibilidade de dependências e alinhamento de documentação de depreciação, com evidências de testes e instalação.

## Escopo obrigatório
1. **Empacotamento**
   - Tornar `pyproject.toml` a fonte canônica de versão/dependências.
   - Sincronizar ou simplificar `setup.cfg` para eliminar drift.
   - Garantir alinhamento com `requirements.txt` (decidir papel: runtime vs dev/ci).

2. **Compatibilidade Python/deps**
   - Revisar `requires-python` e extras com `tensorflow-cpu`.
   - Adicionar marcadores de ambiente quando necessário (ex.: `python_version >= '3.10'`).
   - Validar instalação para 3.8, 3.9, 3.10 e 3.11.

3. **Documentação e contrato de depreciação v1**
   - Alinhar `README.md`, `CHANGELOG.md` e `docs/architecture/specs/PUBLIC_API_V2_CONTRACT.md` ao comportamento real.
   - Se aliases v1 continuarem ativos, ajustar cronograma e critérios documentais.

4. **Configuração de testes**
   - Remover duplicidade de configuração de pytest (`pytest.ini` vs `pyproject.toml`).

## Passo a passo de execução

### CRIAR
- Criar relatório inicial em `docs/development/prompts/relatorios/ISSUE-XXX-auditoria-deps-packaging.md`.
- Criar log de execução em `docs/development/prompts/logs/ISSUE-XXX-auditoria-deps-packaging-LOG.md`.

### ANALISAR
- Mapear divergências atuais entre:
  - `pyproject.toml`
  - `setup.cfg`
  - `requirements.txt`
- Levantar impacto em instalação por versão de Python.

### EXECUTAR
1. Aplicar correções de metadados/dependências.
2. Atualizar documentação afetada.
3. Rodar validações:
   ```bash
   python -m compileall src/pyfolds
   PYTHONPATH=src pytest tests/unit/test_public_import_surface.py -q
   python -m pip install --dry-run -e .
   python -m pip install --dry-run -e '.[dev]'
   ```
4. Repetir instalação em matriz de versões Python (3.8–3.11).

### FINALIZAR
- Registrar evidências no relatório/log.
- Atualizar HUB/fila de execução conforme governança do repositório.
- Abrir PR com:
  - resumo técnico,
  - riscos mitigados,
  - matriz de instalação,
  - testes executados,
  - pendências remanescentes.

## Critérios de aceite
- Sem divergência de versão do pacote entre manifestos.
- Política de suporte Python explícita e coerente com deps.
- Documentação de depreciação consistente com comportamento real.
- Sem warning de configuração duplicada do pytest.
- Instalação (dry-run) validada para versões alvo, com evidência.
