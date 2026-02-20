#!/usr/bin/env python3
"""Validate canonical release version across key files."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

CANONICAL_VERSION = "2.0.3"
REPO_ROOT = Path(__file__).resolve().parents[1]

VERSION_PATTERNS = {
    "src/pyfolds/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "src/pyfolds/core/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "src/pyfolds/layers/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "src/pyfolds/network/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "src/pyfolds/telemetry/__init__.py": r'^__version__\s*=\s*"([^"]+)"',
    "docs/public/guides/engineering_patterns.md": r'version="([^"]+)"',
}


def read_pyproject_version() -> str:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def read_version_with_regex(file_path: str, pattern: str) -> str:
    content = (REPO_ROOT / file_path).read_text(encoding="utf-8")
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find version in {file_path}")
    return match.group(1)


def main() -> int:
    found_versions: dict[str, str] = {"pyproject.toml": read_pyproject_version()}

    for file_path, pattern in VERSION_PATTERNS.items():
        found_versions[file_path] = read_version_with_regex(file_path, pattern)

    mismatches = {
        file_path: version
        for file_path, version in found_versions.items()
        if version != CANONICAL_VERSION
    }

    if mismatches:
        print("❌ Divergent release versions detected:")
        for file_path, version in mismatches.items():
            print(f"  - {file_path}: {version} (expected {CANONICAL_VERSION})")
        return 1

    print(f"✅ All key files are aligned to canonical release version {CANONICAL_VERSION}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
