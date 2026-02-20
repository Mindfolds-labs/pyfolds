# Relatório de Instalação, Análise e Execução em Lote — PyFolds

Data: 2026-02-20  
Projeto: `pyfolds`  
Versão alvo validada: `2.1.0`

## 1) Ordem organizada das etapas solicitadas
1. Levantamento de dependências e aplicações necessárias para execução total dos testes.
2. Análise dos arquivos de instalação para consolidar ambiente de desenvolvimento atualizável.
3. Instalação dos componentes obrigatórios (incluindo Torch, TorchVision e TensorFlow).
4. Análise de código (incluindo backend TensorFlow) e validação de contratos.
5. Execução de testes em lote (suite completa, sem execução isolada como etapa final).
6. Diagnóstico e correção de falhas encontradas.
7. Reexecução em lote para validar estabilização.
8. Geração deste relatório com resultados completos.

## 2) Dependências e aplicações identificadas
### Runtime principal
- torch>=2.0.0
- torchvision>=0.15.0
- numpy>=1.24.0
- zstandard>=0.21.0
- google-crc32c>=1.5.0
- reedsolo>=1.7.0
- safetensors>=0.4.3

### Desenvolvimento e validação
- pytest>=7.0.0
- black>=23.0.0
- isort>=5.0.0
- mypy>=1.0.0

### Stack complementar (full)
- h5py>=3.8.0
- msgpack>=1.0.5
- lz4>=4.3.2
- matplotlib>=3.7.0
- tqdm>=4.65.0

### TensorFlow backend (solicitado)
- tensorflow-cpu>=2.20.0

## 3) Comandos executados (instalação e validação)
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev,full]' tensorflow-cpu
python -m compileall src tests
python tools/check_release_versions.py
PYTHONPATH=src pytest tests -q --maxfail=1 --junitxml=outputs/test_logs/pytest-junit-2.1.0.xml
PYTHONPATH=src pytest tests/unit/advanced/test_backprop.py -q
PYTHONPATH=src pytest tests/unit/core/test_monitoring_and_checkpoint.py::test_model_integrity_monitor_detects_unexpected_mutation tests/unit/core/test_monitoring_and_checkpoint.py::test_model_integrity_monitor_initializes_hash_on_first_check -q
PYTHONPATH=src pytest tests -q --junitxml=outputs/test_logs/pytest-junit-2.1.0.xml
```

## 4) Falhas encontradas e correções aplicadas
### Falha 1 (backprop)
- Sintoma: `test_dendrite_amplification_decay` falhando por ausência de decaimento na primeira janela com evento pendente.
- Causa: `_last_backprop_time` iniciava como `None`, impedindo cálculo de decaimento no primeiro processamento.
- Correção: inicialização para `0.0` no estado de backprop, preservando comportamento temporal esperado pelo contrato.

### Falha 2 (monitor de integridade)
- Sintomas:
  - `ModelIntegrityMonitor` sem método `set_baseline`.
  - payload sem chave `hash_initialized` em `check_integrity`.
- Causa: regressão de compatibilidade após refatoração do monitor para herdar de `WeightIntegrityMonitor` sem manter contrato legado.
- Correção:
  - adicionado `set_baseline()`;
  - implementado `check_integrity()` com payload legado (`integrity_ok`, `current_hash`, `expected_hash`, `hash_initialized`, `step`, `checked`).

## 5) Resultado final da execução em lote
- Coleta: 317 testes (4 desmarcados pelo filtro da suite), 313 executáveis.
- Resultado final: **310 passed, 3 skipped, 4 deselected, 0 failed**.
- Evidência XML (Junit): `outputs/test_logs/pytest-junit-2.1.0.xml`.

## 6) Ajustes de ambiente e versionamento
- Versão do framework atualizada de `2.0.3` para `2.1.0` em metadados canônicos.
- Dependência de TensorFlow integrada para ambiente de desenvolvimento e extra dedicado de instalação.
- Arquivos de instalação/documentação alinhados para atualização futura sem divergência de versão.
