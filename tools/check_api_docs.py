#!/usr/bin/env python3
"""Validate docstrings for public API symbols.

By default, reports missing docstrings and exits 0.
With --strict, exits 1 when missing docstrings are found.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_ROOT = Path("src/pyfolds")


def iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.name != "__init__.py")


def missing_public_docstrings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    missing: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            if ast.get_docstring(node):
                continue
            missing.append(node.name)

    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public API docstrings")
    parser.add_argument("--strict", action="store_true", help="Fail on missing docstrings")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Path root to scan (default: src/pyfolds)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Path not found: {args.root}")
        return 2

    failures: list[tuple[Path, str]] = []
    for file_path in iter_python_files(args.root):
        for symbol in missing_public_docstrings(file_path):
            failures.append((file_path, symbol))

    if failures:
        print("Missing docstrings in public symbols:")
        for file_path, symbol in failures:
            print(f"- {file_path}: {symbol}")
        if args.strict:
            return 1

    print("Public API docstring check completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
