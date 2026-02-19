#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORTS_DIR="docs/sheer-audit/reports"
DATA_DIR="docs/sheer-audit/data"
UML_DIR="${DATA_DIR}/uml"
SHEERDOCS_DIR="docs/sheer-audit/sheerdocs"
LOG_FILE="${REPORTS_DIR}/sheer_audit_run_${TIMESTAMP}.log"

mkdir -p "${REPORTS_DIR}" "${DATA_DIR}" "${UML_DIR}" "${SHEERDOCS_DIR}"

exec > >(tee "${LOG_FILE}") 2>&1

echo "[INFO] Iniciando run_sheer_audit.sh"
echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Log: ${LOG_FILE}"

fail() {
  echo "[ERROR] $*"
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "Arquivo obrigatório não encontrado: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || fail "Diretório obrigatório não encontrado: $path"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || fail "Dependência obrigatória ausente: $cmd"
}

validate_prereqs() {
  echo "[INFO] Validando pré-requisitos"

  require_cmd python3
  require_file "docs/sheer-audit/sheer.toml"
  require_file "docs/sheer-audit/scripts/generate_repo_model.py"
  require_dir "src"

  python3 - <<'PY'
import importlib
import os
import tomllib
from pathlib import Path

cfg_path = Path("docs/sheer-audit/sheer.toml")
config = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

for dep in ("sheer_audit",):
    try:
        importlib.import_module(dep)
    except Exception as exc:
        raise SystemExit(f"Dependência Python ausente: {dep} ({exc})")

root = Path(config.get("project", {}).get("root", ".")).resolve()
if not root.exists():
    raise SystemExit(f"project.root inválido no sheer.toml: {root}")

scan = config.get("scan", {})
for d in scan.get("include_dirs", []):
    p = (Path.cwd() / d).resolve()
    if not p.exists():
        print(f"[WARN] include_dir inexistente: {d} -> {p}")

uml_out = config.get("uml", {}).get("output_dir", "docs/sheer-audit/data/uml")
Path(uml_out).mkdir(parents=True, exist_ok=True)

print("Pré-requisitos Python e configuração OK")
PY

  echo "[INFO] Pré-requisitos validados"
}

generate_outputs() {
  echo "[INFO] Executando generate_repo_model.py"
  python3 docs/sheer-audit/scripts/generate_repo_model.py

  require_file "docs/sheer-audit/data/repo_model.json"

  echo "[INFO] Gerando code_map.{json,md} e UML .puml"
  python3 - <<'PY'
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

base = Path("docs/sheer-audit/data")
repo_model = json.loads((base / "repo_model.json").read_text(encoding="utf-8"))
symbols = repo_model.get("symbols", [])
edges = repo_model.get("edges", [])

modules = {}
classes = {}
functions = {}
methods_by_class = defaultdict(list)
imports_by_module = defaultdict(set)
contains = defaultdict(list)

for s in symbols:
    sid = s.get("id", "")
    kind = s.get("kind")
    if kind == "module":
        modules[sid] = s
    elif kind == "class":
        classes[sid] = s
    elif kind == "function":
        functions[sid] = s
    elif kind == "method":
        methods_by_class[sid.rsplit(".", 1)[0]].append(s)

for e in edges:
    if e.get("type") == "IMPORT":
        imports_by_module[e.get("src", "")].add(e.get("dst", ""))
    elif e.get("type") == "CONTAINS":
        contains[e.get("src", "")].append(e.get("dst", ""))

code_map = {
    "repo": repo_model.get("repo", {}),
    "metrics": repo_model.get("metrics", {}),
    "modules": [],
}

for mod_id in sorted(modules.keys()):
    m = modules[mod_id]
    module_item = {
        "id": mod_id,
        "qname": m.get("qname"),
        "file": m.get("file"),
        "imports": sorted(imports_by_module.get(mod_id, [])),
        "classes": [],
        "functions": [],
    }

    for child_id in sorted(contains.get(mod_id, [])):
        if child_id in classes:
            c = classes[child_id]
            class_item = {
                "id": child_id,
                "name": c.get("name"),
                "qname": c.get("qname"),
                "line": c.get("line"),
                "bases": c.get("bases", []),
                "methods": [],
            }
            for mtd_id in sorted(contains.get(child_id, [])):
                class_item["methods"].append(mtd_id)
            module_item["classes"].append(class_item)
        elif child_id in functions:
            f = functions[child_id]
            module_item["functions"].append(
                {
                    "id": child_id,
                    "name": f.get("name"),
                    "qname": f.get("qname"),
                    "line": f.get("line"),
                    "params": f.get("params", []),
                }
            )

    code_map["modules"].append(module_item)

(base / "code_map.json").write_text(json.dumps(code_map, indent=2, ensure_ascii=False), encoding="utf-8")

md_lines = [
    "# Code Map (Sheer Audit)",
    "",
    f"- Repositório: `{code_map['repo'].get('name', 'unknown')}`",
    f"- Arquivos Python: `{code_map['metrics'].get('python_files', 0)}`",
    f"- Símbolos: `{code_map['metrics'].get('symbols', 0)}`",
    "",
    "## Módulos",
    "",
]

for mod in code_map["modules"]:
    md_lines.append(f"### `{mod['qname']}`")
    md_lines.append(f"- Arquivo: `{mod['file']}`")
    if mod["imports"]:
        md_lines.append("- Imports:")
        for imp in mod["imports"][:30]:
            md_lines.append(f"  - `{imp}`")
    if mod["classes"]:
        md_lines.append("- Classes:")
        for c in mod["classes"]:
            bases = ", ".join(c.get("bases", [])) or "(sem base explícita)"
            md_lines.append(f"  - `{c['name']}` (bases: {bases})")
    if mod["functions"]:
        md_lines.append("- Funções:")
        for f in mod["functions"]:
            md_lines.append(f"  - `{f['name']}({', '.join(f.get('params', []))})`")
    md_lines.append("")

(base / "code_map.md").write_text("\n".join(md_lines), encoding="utf-8")

uml_dir = base / "uml"
uml_dir.mkdir(parents=True, exist_ok=True)

pkg_lines = ["@startuml", "skinparam packageStyle rectangle", ""]
for mod in code_map["modules"]:
    pkg_lines.append(f"package \"{mod['qname']}\" {{}}")
pkg_lines.append("")
for mod in code_map["modules"]:
    src = mod["qname"]
    for imp in mod["imports"]:
        if imp.startswith("mod:"):
            dst = imp.removeprefix("mod:")
            if dst.startswith("src.pyfolds"):
                pkg_lines.append(f"\"{src}\" --> \"{dst}\"")
pkg_lines.append("@enduml")
(uml_dir / "package.puml").write_text("\n".join(pkg_lines), encoding="utf-8")

class_lines = ["@startuml", "hide empty members", ""]
for mod in code_map["modules"]:
    for c in mod["classes"]:
        cid = c["qname"]
        class_lines.append(f"class \"{cid}\" {{")
        for mtd in c.get("methods", []):
            mname = mtd.split(".")[-1]
            class_lines.append(f"  +{mname}()")
        class_lines.append("}")
    class_lines.append("")
(uml_dir / "class_overview.puml").write_text("\n".join(class_lines + ["@enduml"]), encoding="utf-8")

print("Gerados: code_map.json, code_map.md, uml/package.puml, uml/class_overview.puml")
PY

  require_file "docs/sheer-audit/data/code_map.json"
  require_file "docs/sheer-audit/data/code_map.md"
  require_file "docs/sheer-audit/data/uml/package.puml"
  require_file "docs/sheer-audit/data/uml/class_overview.puml"
}

render_uml_if_possible() {
  echo "[INFO] Verificando renderização UML"
  if command -v plantuml >/dev/null 2>&1; then
    plantuml -tsvg "${UML_DIR}/package.puml" "${UML_DIR}/class_overview.puml" || fail "Falha ao renderizar UML com plantuml"
    echo "[INFO] UML renderizada com plantuml"
  else
    echo "[WARN] plantuml não encontrado; mantendo apenas .puml"
  fi
}

check_critical_architecture_violations() {
  echo "[INFO] Avaliando violações arquiteturais críticas"
  python3 - <<'PY'
from __future__ import annotations

import json
import tomllib
from fnmatch import fnmatch
from pathlib import Path

repo_model_path = Path("docs/sheer-audit/data/repo_model.json")
config_path = Path("docs/sheer-audit/sheer.toml")

repo_model = json.loads(repo_model_path.read_text(encoding="utf-8"))
config = tomllib.loads(config_path.read_text(encoding="utf-8"))

architecture_cfg = config.get("architecture", {})
layers = architecture_cfg.get("layers", [])
layer_paths = architecture_cfg.get("layer_paths", ["src/pyfolds"])
forbidden_imports = architecture_cfg.get("forbidden_imports", [])
enforce_layering = architecture_cfg.get("enforce_layering", False)

layer_index = {layer: idx for idx, layer in enumerate(layers)}
module_to_layer: dict[str, str] = {}

for symbol in repo_model.get("symbols", []):
    if symbol.get("kind") != "module":
        continue

    qname = symbol.get("qname") or ""
    file_path = symbol.get("file") or ""
    if not qname or not file_path:
        continue

    if not any(file_path.startswith(prefix) for prefix in layer_paths):
        continue

    parts = qname.split(".")
    layer_name = parts[2] if len(parts) > 2 and parts[1] == "pyfolds" else (parts[1] if len(parts) > 1 else "")
    if layer_name in layer_index:
        module_to_layer[qname] = layer_name

critical_violations: list[str] = []
for edge in repo_model.get("edges", []):
    if edge.get("type") != "IMPORT":
        continue

    src = (edge.get("src") or "").removeprefix("mod:")
    dst = (edge.get("dst") or "").removeprefix("mod:")
    if not src or not dst:
        continue

    if src.startswith("src.pyfolds") and dst.startswith("tests"):
        critical_violations.append(f"Import proibido de testes em código fonte: {src} -> {dst}")

    for rule in forbidden_imports:
        src_pattern = (rule.get("src") or "*").removeprefix("mod:")
        dst_pattern = (rule.get("dst") or "*").removeprefix("mod:")
        severity = (rule.get("severity") or "critical").lower()
        if severity == "critical" and fnmatch(src, src_pattern) and fnmatch(dst, dst_pattern):
            critical_violations.append(
                f"Regra forbidden_imports violada ({src_pattern} -> {dst_pattern}): {src} -> {dst}"
            )

    if enforce_layering:
        src_layer = module_to_layer.get(src)
        dst_layer = module_to_layer.get(dst)
        if src_layer and dst_layer and layer_index[src_layer] < layer_index[dst_layer]:
            critical_violations.append(
                f"Dependência invertida entre camadas ({src_layer} -> {dst_layer}): {src} -> {dst}"
            )

if critical_violations:
    print("[CRITICAL] Violações arquiteturais detectadas:")
    for violation in sorted(set(critical_violations)):
        print(f" - {violation}")
    raise SystemExit(1)

print("[INFO] Nenhuma violação arquitetural crítica detectada")
PY
}

sync_sheerdocs() {
  echo "[INFO] Sincronizando artefatos para ${SHEERDOCS_DIR}"
  mkdir -p "${SHEERDOCS_DIR}/uml"

  cp -f "${DATA_DIR}/repo_model.json" "${SHEERDOCS_DIR}/repo_model.json"
  cp -f "${DATA_DIR}/code_map.json" "${SHEERDOCS_DIR}/code_map.json"
  cp -f "${DATA_DIR}/code_map.md" "${SHEERDOCS_DIR}/code_map.md"
  cp -f "${UML_DIR}/package.puml" "${SHEERDOCS_DIR}/uml/package.puml"
  cp -f "${UML_DIR}/class_overview.puml" "${SHEERDOCS_DIR}/uml/class_overview.puml"

  for rendered in "${UML_DIR}"/*.svg "${UML_DIR}"/*.png; do
    if [[ -f "${rendered}" ]]; then
      cp -f "${rendered}" "${SHEERDOCS_DIR}/uml/"
    fi
  done

  cp -f "${LOG_FILE}" "${SHEERDOCS_DIR}/"
  echo "[INFO] Sync concluído"
}

validate_prereqs
generate_outputs
render_uml_if_possible
check_critical_architecture_violations
sync_sheerdocs

echo "[INFO] Execução concluída com sucesso"
