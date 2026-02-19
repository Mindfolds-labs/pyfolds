#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil

try:
    import cairosvg
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with: pip install cairosvg"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "brand" / "pyfolds-icon-master.svg"
OUT = ROOT / "_static" / "brand"
PKG_OUT = ROOT.parent / "src" / "pyfolds" / "assets" / "brand"
RASTER_OUT = ROOT / "_build" / "brand"

VARIANTS = {
    "dark": {},
    "light": {
        "#08142D": "#F4F8FF",
        "#102B5F": "#DCE9FF",
        "#143574": "#2D4A85",
        "#2B63D6": "#3563CC",
        "#183B86": "#224588",
    },
    "mono": {
        "#08142D": "#111111",
        "#102B5F": "#333333",
        "#3B7BFF": "#555555",
        "#1C3F8E": "#444444",
        "#143574": "#777777",
        "#2B63D6": "#666666",
        "#183B86": "#888888",
        "#FFD86A": "#999999",
        "#E5A800": "#555555",
    },
}

SIZES = [16, 32, 48, 64, 128, 256, 512]


def render_svg(source: str, target: Path) -> None:
    target.write_text(source, encoding="utf-8")


def render_png(svg_path: Path, png_path: Path, size: int) -> None:
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), output_width=size, output_height=size)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PKG_OUT.mkdir(parents=True, exist_ok=True)
    RASTER_OUT.mkdir(parents=True, exist_ok=True)

    master = MASTER.read_text(encoding="utf-8")

    for variant, replacements in VARIANTS.items():
        content = master
        for src, dst in replacements.items():
            content = content.replace(src, dst)
        svg_path = OUT / f"pyfolds-icon-{variant}.svg"
        render_svg(content, svg_path)

        for size in SIZES:
            render_png(svg_path, RASTER_OUT / f"pyfolds-icon-{variant}-{size}.png", size)

    shutil.copy2(OUT / "pyfolds-icon-dark.svg", OUT / "favicon.svg")
    shutil.copy2(OUT / "pyfolds-icon-dark.svg", PKG_OUT / "pyfolds-icon.svg")


if __name__ == "__main__":
    main()
