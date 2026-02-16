#!/usr/bin/env python3
"""Validate local markdown links (files + anchors) in docs and README."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADER_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def slugify(title: str) -> str:
    slug = title.strip().lower()
    slug = re.sub(r"[`*_~]", "", slug)
    slug = re.sub(r"[^\w\-\s]", "", slug)
    slug = slug.replace(" ", "-")
    slug = re.sub(r"-+", "-", slug)
    return slug


def parse_anchors(md_path: Path) -> set[str]:
    anchors: set[str] = set()
    content = md_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        match = HEADER_RE.match(line)
        if match:
            anchors.add(slugify(match.group(1)))
    return anchors


def ignored(target: str) -> bool:
    raw = target.strip()
    return (
        not raw
        or raw.startswith("http://")
        or raw.startswith("https://")
        or raw.startswith("mailto:")
        or raw.startswith("data:")
    )


def split_target(raw: str) -> tuple[str, str | None]:
    cleaned = raw.strip().split()[0]
    if "#" in cleaned:
        file_part, anchor = cleaned.split("#", 1)
        return file_part, anchor or None
    return cleaned, None


def validate_markdown_file(md_file: Path) -> list[str]:
    normalized = str(md_file).replace("\\", "/")
    if "docs/governance/adr" in normalized:
        return []

    errors: list[str] = []
    content = md_file.read_text(encoding="utf-8")

    for target_raw in LINK_RE.findall(content):
        if ignored(target_raw):
            continue

        rel_path, anchor = split_target(target_raw)

        if rel_path == "":
            target_file = md_file
        else:
            target_file = (md_file.parent / rel_path).resolve()
            if not target_file.exists():
                errors.append(f"{md_file}: missing path '{rel_path}'")
                continue

        if anchor:
            if target_file.suffix.lower() != ".md":
                continue
            anchors = parse_anchors(target_file)
            if slugify(anchor) not in anchors:
                errors.append(f"{md_file}: missing anchor '#{anchor}' in '{target_file}'")

    return errors


def collect_markdown_paths(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.is_file() and path.suffix.lower() == ".md":
            files.append(path)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local markdown links")
    parser.add_argument("paths", nargs="*", default=["docs", "README.md"])
    args = parser.parse_args()

    markdown_files = collect_markdown_paths(args.paths)
    if not markdown_files:
        print("No markdown files found to scan.", file=sys.stderr)
        return 2

    failures: list[str] = []
    for md_file in markdown_files:
        failures.extend(validate_markdown_file(md_file))

    if failures:
        print("Broken markdown links found:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"OK: validated {len(markdown_files)} markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
