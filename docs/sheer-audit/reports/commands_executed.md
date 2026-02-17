# Comandos executados

```bash
git clone https://github.com/Mindfolds-labs/sheer-audit.git /workspace/sheer-audit
cd /workspace/sheer-audit && python -m pip install -e .
cd /workspace/sheer-audit && pytest -q
cd /workspace/pyfolds && python - <<'PY'
from sheer_audit.config import ScanConfig
from sheer_audit.scan.repo import collect_python_files
# geração dos JSONs em docs/sheer-audit/data
PY
```

## Estado da CLI

Comando observado:

```bash
sheer --help
```

Saída:

```text
Sheer Audit instalado com sucesso!
Execute: sheer --help
```

A CLI instalada nesta versão não expõe subcomandos funcionais de auditoria diretamente.
