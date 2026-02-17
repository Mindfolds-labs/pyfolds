from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

from sheer_audit.config import load_config
from sheer_audit.model.schema import Edge, Finding, RepoInfo, RepoModel, Symbol
from sheer_audit.scan.repo import collect_python_files


def module_name_from_path(path: str) -> str:
    p = Path(path)
    if p.suffix == ".py":
        p = p.with_suffix("")
    mod = p.as_posix().replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def main() -> None:
    repo_root = Path(".").resolve()
    config = load_config("docs/sheer-audit/sheer.toml")
    files = collect_python_files(str(repo_root), config.scan)

    symbols: List[Symbol] = []
    edges: List[Edge] = []
    findings: List[Finding] = []

    module_symbol_ids: Dict[str, str] = {}
    class_symbol_ids: Dict[Tuple[str, str], str] = {}

    for rel in files:
        rel_path = Path(rel)
        module_qname = module_name_from_path(rel)
        module_symbol_id = f"mod:{module_qname}"
        module_symbol_ids[module_qname] = module_symbol_id

        symbols.append(
            Symbol(
                id=module_symbol_id,
                kind="module",
                name=module_qname.split(".")[-1],
                qname=module_qname,
                file=rel,
                line=1,
            )
        )

        full_path = repo_root / rel_path
        try:
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError as exc:
            findings.append(
                Finding(
                    code="PARSE_ERROR",
                    severity="ERROR",
                    file=rel,
                    line=exc.lineno,
                    column=exc.offset,
                    message=str(exc),
                )
            )
            continue

        for node in tree.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    imported = name.name
                    edges.append(Edge(type="IMPORT", src=module_symbol_id, dst=f"mod:{imported}"))
            elif isinstance(node, ast.ImportFrom):
                imported_mod = node.module or ""
                if node.level:
                    parts = module_qname.split(".")
                    prefix = ".".join(parts[: len(parts) - node.level])
                    imported_mod = f"{prefix}.{imported_mod}".strip(".")
                if imported_mod:
                    edges.append(Edge(type="IMPORT", src=module_symbol_id, dst=f"mod:{imported_mod}"))
            elif isinstance(node, ast.ClassDef):
                class_qname = f"{module_qname}.{node.name}"
                class_id = f"cls:{class_qname}"
                class_symbol_ids[(module_qname, node.name)] = class_id
                symbols.append(
                    Symbol(
                        id=class_id,
                        kind="class",
                        name=node.name,
                        qname=class_qname,
                        file=rel,
                        line=node.lineno,
                        doc=ast.get_docstring(node),
                        bases=[ast.unparse(b) for b in node.bases],
                        decorators=[ast.unparse(d) for d in node.decorator_list],
                    )
                )
                edges.append(Edge(type="CONTAINS", src=module_symbol_id, dst=class_id))

                for base in node.bases:
                    if isinstance(base, ast.Name):
                        edges.append(Edge(type="INHERITS", src=class_id, dst=f"cls:*.{base.id}"))
                    elif isinstance(base, ast.Attribute):
                        edges.append(Edge(type="INHERITS", src=class_id, dst=f"cls:{ast.unparse(base)}"))

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_qname = f"{class_qname}.{child.name}"
                        method_id = f"mtd:{method_qname}"
                        symbols.append(
                            Symbol(
                                id=method_id,
                                kind="method",
                                name=child.name,
                                qname=method_qname,
                                file=rel,
                                line=child.lineno,
                                doc=ast.get_docstring(child),
                                params=[a.arg for a in child.args.args],
                                decorators=[ast.unparse(d) for d in child.decorator_list],
                            )
                        )
                        edges.append(Edge(type="CONTAINS", src=class_id, dst=method_id))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_qname = f"{module_qname}.{node.name}"
                fn_id = f"fn:{fn_qname}"
                symbols.append(
                    Symbol(
                        id=fn_id,
                        kind="function",
                        name=node.name,
                        qname=fn_qname,
                        file=rel,
                        line=node.lineno,
                        doc=ast.get_docstring(node),
                        params=[a.arg for a in node.args.args],
                        decorators=[ast.unparse(d) for d in node.decorator_list],
                    )
                )
                edges.append(Edge(type="CONTAINS", src=module_symbol_id, dst=fn_id))

    internal_imports = [e for e in edges if e.type == "IMPORT" and e.dst.startswith("mod:src.pyfolds")]

    model = RepoModel(
        repo=RepoInfo(root=str(repo_root), name="pyfolds"),
        symbols=symbols,
        edges=edges,
        findings=findings,
        metrics={
            "python_files": len(files),
            "symbols": len(symbols),
            "edges": len(edges),
            "findings": len(findings),
            "internal_import_edges": len(internal_imports),
        },
    )

    out = Path("docs/sheer-audit/data/repo_model.json")
    out.write_text(json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
