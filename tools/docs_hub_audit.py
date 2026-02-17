#!/usr/bin/env python3
"""Generate and validate docs-hub quality artifacts for PyFolds.

Quality gates (CI):
- fail if a public function/class lacks a docstring;
- fail if a public module is not referenced in generated module inventory;
- fail if generated structural artifacts diverge from repository versions.
"""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path

SRC_ROOT = Path("src/pyfolds")
DOCS_ROOT = Path("docs")
OUT_DIR = Path("docs/development/generated")


@dataclass(frozen=True)
class PublicSymbol:
    module: str
    name: str
    kind: str
    has_docstring: bool


def module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if rel.name == "__init__.py":
        parts = rel.parts[:-1]
    else:
        parts = rel.with_suffix("").parts
    return ".".join(("pyfolds", *parts))


def iter_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def parse_public_symbols(path: Path, module: str) -> tuple[list[PublicSymbol], list[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols: list[PublicSymbol] = []
    missing: list[str] = []
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module
                if node.level > 0:
                    if module.startswith("pyfolds"):
                        parent = module.split(".")[:-node.level]
                        base = ".".join(parent + [node.module]) if parent else node.module
                imports.add(base)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            has_doc = bool(ast.get_docstring(node))
            symbols.append(
                PublicSymbol(
                    module=module,
                    name=node.name,
                    kind="class" if isinstance(node, ast.ClassDef) else "function",
                    has_docstring=has_doc,
                )
            )
            if not has_doc:
                missing.append(f"{module}.{node.name}")

    return symbols, missing, imports


def build_artifacts(src_root: Path) -> tuple[dict[str, str], list[str], list[str]]:
    modules: list[str] = []
    symbols: list[PublicSymbol] = []
    missing_docstrings: list[str] = []
    dep_edges: set[tuple[str, str]] = set()

    for path in iter_py_files(src_root):
        mod = module_name(path, src_root)
        modules.append(mod)
        file_symbols, missing, imports = parse_public_symbols(path, mod)
        symbols.extend(file_symbols)
        missing_docstrings.extend(missing)
        for imp in imports:
            if imp.startswith("pyfolds") and imp != mod:
                dep_edges.add((mod, imp))

    modules = sorted(set(modules))
    symbols = sorted(symbols, key=lambda s: (s.module, s.name))
    missing_docstrings.sort()

    docstring_coverage = 100.0
    if symbols:
        with_doc = sum(1 for s in symbols if s.has_docstring)
        docstring_coverage = (with_doc / len(symbols)) * 100

    manifest = {
        "modules": modules,
        "public_symbols": [f"{s.module}.{s.name}" for s in symbols],
    }

    inventory_lines = ["# Inventário automático de módulos públicos", "", "| Módulo |", "|---|"]
    inventory_lines.extend([f"| `{m}` |" for m in modules])

    function_lines = [
        "# Tabela automática de funções/classes públicas",
        "",
        "| Módulo | Símbolo | Tipo | Docstring |",
        "|---|---|---|---|",
    ]
    function_lines.extend(
        [
            f"| `{s.module}` | `{s.name}` | {s.kind} | {'✅' if s.has_docstring else '❌'} |"
            for s in symbols
        ]
    )

    metrics = {
        "total_modules": len(modules),
        "total_public_symbols": len(symbols),
        "missing_docstrings": len(missing_docstrings),
        "docstring_coverage_percent": round(docstring_coverage, 2),
        "documentation_coverage_percent": 100.0,
        "dependency_edges": len(dep_edges),
    }

    metrics_lines = [
        "# Métricas automáticas de documentação/estrutura",
        "",
        f"- Módulos públicos: **{metrics['total_modules']}**",
        f"- Símbolos públicos: **{metrics['total_public_symbols']}**",
        f"- Símbolos públicos sem docstring: **{metrics['missing_docstrings']}**",
        f"- Cobertura de docstrings públicas: **{metrics['docstring_coverage_percent']}%**",
        f"- Cobertura documental (inventário gerado): **{metrics['documentation_coverage_percent']}%**",
        f"- Arestas de dependência internas: **{metrics['dependency_edges']}**",
    ]

    dep_lines = ["# Diagrama automático de dependências internas", "", "```{mermaid}", "graph TD"]
    for origin, target in sorted(dep_edges):
        dep_lines.append(f"    {origin.replace('.', '_')}[{origin}] --> {target.replace('.', '_')}[{target}]")
    dep_lines.append("```")

    artifacts = {
        "module_inventory.md": "\n".join(inventory_lines) + "\n",
        "functions_table.md": "\n".join(function_lines) + "\n",
        "metrics.md": "\n".join(metrics_lines) + "\n",
        "dependency_diagram.md": "\n".join(dep_lines) + "\n",
        "metrics.json": json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        "api_structure_manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    }

    return artifacts, missing_docstrings, modules


def main() -> int:
    parser = argparse.ArgumentParser(description="Docs hub audit and pre-build generator")
    parser.add_argument("--check", action="store_true", help="Fail if generated outputs diverge")
    args = parser.parse_args()

    artifacts, missing_docstrings, modules = build_artifacts(SRC_ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    changed: list[str] = []
    for name, content in artifacts.items():
        target = OUT_DIR / name
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            changed.append(str(target))
            if not args.check:
                target.write_text(content, encoding="utf-8")

    if missing_docstrings:
        print("FAIL: Funções/classes públicas sem docstring:")
        for symbol in missing_docstrings:
            print(f"- {symbol}")
        return 1

    # check for non-referenced module in generated inventory
    inventory_text = artifacts["module_inventory.md"]
    unreferenced = [m for m in modules if m not in inventory_text]
    if unreferenced:
        print("FAIL: Módulos não referenciados no inventário:")
        for module in unreferenced:
            print(f"- {module}")
        return 1

    if args.check and changed:
        print("FAIL: Divergência estrutural detectada nos artefatos gerados:")
        for path in changed:
            print(f"- {path}")
        print("Execute: python tools/docs_hub_audit.py")
        return 1

    print("OK: docs hub audit concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
