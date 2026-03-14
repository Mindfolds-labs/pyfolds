# Relatório Técnico de Auditoria — PyFolds

## 1) Governança aplicada (CRIAR → ANALISAR → EXECUTAR → FINALIZAR)

### CRIAR
- Escopo definido para auditoria estática + instalação multi-versão Python + análise de riscos.
- Artefatos criados na **raiz** do repositório:
  - `RELATORIO_AUDITORIA_TECNICA_PYFOLDS.md` (este relatório)
  - `PROMPT_CODEX_PROXIMA_EXECUCAO.md` (prompt operacional para próxima execução)

### ANALISAR

#### 2.1 Estrutura e qualidade do código
- Estrutura principal está organizada com foco em pacote moderno (`src/pyfolds`), suíte de testes extensa (`tests`) e scripts operacionais (`scripts`).
- Métricas rápidas de estrutura:
  - `src/pyfolds`: 114 arquivos `.py`
  - `tests`: 128 arquivos `test_*.py`
  - `scripts`: 2 scripts Python principais + runners auxiliares shell/ps1
- Qualidade estrutural observada:
  - Boa modularização por domínio (core, advanced, serialization, telemetry, etc.).
  - Cobertura de testes distribuída por áreas (unit/integration/performance/serialization/telemetry).
  - Há coexistência de `pytest.ini` e configuração de pytest no `pyproject.toml`, gerando warning de ignorar seção no pyproject.

#### 2.2 Conformidade de dependências

##### Divergências encontradas entre manifestos
- **Inconsistência de versão do pacote**:
  - `pyproject.toml`: `2.2.0`
  - `setup.cfg`: `2.1.0`
- **Dependências não alinhadas**:
  - `setup.cfg` e `requirements.txt` incluem `torchvision` e `safetensors` no runtime, mas `pyproject.toml` não inclui no bloco principal de `dependencies`.
  - `requirements.txt` inclui `tensorflow-cpu>=2.20.0` em runtime base, enquanto no `pyproject.toml` TensorFlow está em extras (`dev`, `tensorflow`, `interop`).

##### Compatibilidade Python
- O projeto declara `requires-python = ">=3.8"`, porém alguns pacotes modernos em versões mais novas já exigem Python >=3.9 ou >=3.10.
- Como os requisitos são com limites **mínimos** (`>=`), a resolução pode buscar versões antigas compatíveis com Python 3.8, mas há risco de resolução frágil com extras pesados.
- Risco concreto para Python 3.8:
  - extras `dev/tensorflow/interop` pedem `tensorflow-cpu>=2.20.0`; versões atuais exigem Python mais novo, podendo quebrar instalação em 3.8.

#### 2.3 Documentação técnica
- `README.md` está detalhado e inclui instalação, quickstart e política de depreciação.
- `CHANGELOG.md` existe e segue formato estruturado.
- Documentação em `docs/` é ampla, incluindo arquitetura e contrato de API pública.
- **Inconsistência de narrativa de depreciação v1**:
  - README e contrato mencionam remoção de aliases v1 em `2.0.0`, mas testes atuais ainda mostram aliases ativos com `DeprecationWarning`.
  - Isso indica possível desalinhamento entre cronograma documentado e estado real do código.

#### 2.4 Riscos priorizados
1. **Risco de publicação/empacotamento**: metadados conflitantes (`pyproject` vs `setup.cfg`) podem causar release inconsistente.
2. **Risco de instalação em Python 3.8**: extras com TensorFlow podem falhar por incompatibilidade de marcador de Python.
3. **Risco de comunicação técnica**: política de depreciação com datas/major version possivelmente defasadas em relação ao comportamento atual.
4. **Risco operacional de QA**: warning de configuração duplicada de pytest pode gerar confusão em CI.

### EXECUTAR

## 3) Evidências de execução (comandos)

### 3.1 Sanidade estática
- `python -m compileall src/pyfolds`
  - Resultado: **sucesso** (compilação completa sem erro sintático).

### 3.2 Teste de unidade smoke
- `PYTHONPATH=src pytest tests/unit/test_public_import_surface.py -q`
  - Resultado: **4 passed**.
  - Observado: warnings esperados de depreciação para aliases v1.

### 3.3 Simulação de instalação em ambiente limpo (venv)
Comando-base aplicado por versão:
```bash
python -m venv /tmp/pyfolds_venv_XX
source /tmp/pyfolds_venv_XX/bin/activate
python -m pip install --upgrade pip
python -m pip install --dry-run -e .
python -m pip install --dry-run -e '.[dev]'
```

Matriz executada:
- Python 3.8: indisponível no ambiente atual (não foi possível simular localmente).
- Python 3.9: indisponível no ambiente atual (não foi possível simular localmente).
- Python 3.10: `-e .` **OK** | `-e '.[dev]'` **OK** (dry-run).
- Python 3.11: `-e .` **OK** | `-e '.[dev]'` **OK** (dry-run).

### FINALIZAR

## 4) Recomendações objetivas (diretas na raiz)

1. **Unificar fonte de verdade do empacotamento**
   - Priorizar `pyproject.toml` como canônico.
   - Sincronizar/remover campos redundantes em `setup.cfg` para evitar drift de versão/dependências.

2. **Revisar política de suporte de Python**
   - Se o objetivo real incluir TensorFlow em workflows oficiais, considerar elevar baseline para `>=3.9` ou `>=3.10`.
   - Alternativamente, separar extras por marcador de versão (`; python_version >= '3.10'`).

3. **Corrigir inconsistência de depreciação v1**
   - Atualizar README/CHANGELOG/contrato para refletir estado atual real (ou executar remoção de aliases conforme documentação).

4. **Eliminar warning de configuração pytest**
   - Consolidar configuração em `pytest.ini` **ou** `pyproject.toml`, evitando duplicidade.

5. **Adicionar job de CI de matriz de instalação**
   - `3.8, 3.9, 3.10, 3.11` com `pip install -e .` e extras críticos (`dev`, `serialization`, `telemetry`, `tensorflow`).

## 5) Falhas encontradas e sugestão de melhoria (resumo executivo)
- Falha de cobertura de auditoria local para 3.8/3.9 por indisponibilidade dos bins no ambiente.
  - Sugestão: pipeline CI dedicado com pyenv/actions para todas as versões alvo.
- Inconsistência de metadados entre manifestos.
  - Sugestão: governança de release com validação automática de sincronismo de versão/deps.
- Contradição entre política de depreciação e prática.
  - Sugestão: ADR curto de alinhamento + atualização documental/contract tests.
