#!/usr/bin/env python3
"""Gera dashboard HTML simples a partir do HUB e métricas."""

from __future__ import annotations

import json
from pathlib import Path

HUB_PATH = Path("docs/development/HUB_CONTROLE.md")
METRICS_PATH = Path("docs/development/generated/metrics.json")
OUTPUT_PATH = Path("docs/development/generated/dashboard.html")


def load_hub_markdown() -> str:
    return HUB_PATH.read_text(encoding="utf-8") if HUB_PATH.exists() else ""


def parse_metrics_json() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def generate_html_dashboard() -> str:
    hub = load_hub_markdown()
    metrics = parse_metrics_json()
    done = hub.count("Concluída")
    planned = hub.count("Planejada")
    html = f"""<!doctype html>
<html lang='pt-BR'><head><meta charset='utf-8'><title>Dashboard HUB</title>
<style>body{{font-family:Arial;margin:2rem}}.kpi{{display:inline-block;margin-right:1rem;padding:1rem;border:1px solid #ccc}}</style>
</head><body>
<h1>Dashboard HUB</h1>
<div class='kpi'>Concluídas: <b>{done}</b></div>
<div class='kpi'>Planejadas: <b>{planned}</b></div>
<pre>{json.dumps(metrics, ensure_ascii=False, indent=2)}</pre>
</body></html>"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    generate_html_dashboard()
    print(OUTPUT_PATH)
