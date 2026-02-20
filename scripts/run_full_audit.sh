#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_step() {
  local name="$1"
  shift
  echo "\n==== [AUDIT] $name ===="
  "$@"
}

run_step "Bytecode/import syntax check" python -m compileall -q src tests run_pyfolds.py

run_step "Public import surface check" python - <<'PY'
import pyfolds
missing = [name for name in pyfolds.__all__ if not hasattr(pyfolds, name)]
if missing:
    raise SystemExit(f"Missing exports in pyfolds.__all__: {missing}")
print(f"public_import_surface_ok: {len(pyfolds.__all__)} exports")
PY

echo "\n==== [AUDIT] Installation smoke test ===="
install_log="$(mktemp)"
if python test_install.py >"$install_log" 2>&1; then
  cat "$install_log"
else
  cat "$install_log"
  if rg -q "Cannot connect to proxy|No matching distribution found for torch|Tunnel connection failed" "$install_log"; then
    echo "⚠️  Installation smoke test skipped due to external network/proxy limitation."
  else
    echo "❌ Installation smoke test failed for a functional reason."
    rm -f "$install_log"
    exit 1
  fi
fi
rm -f "$install_log"

run_step "Full test suite" pytest -q

echo "\n✅ Auditoria completa concluída com sucesso."
