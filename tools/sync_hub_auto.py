#!/usr/bin/env python3
"""Sincroniza cards do HUB a partir da fila CSV."""

from __future__ import annotations

from pathlib import Path

try:
    from tools import sync_hub
except ImportError:
    import sync_hub

CSV_PATH = Path("docs/development/execution_queue.csv")
HUB_PATH = Path("docs/development/HUB_CONTROLE.md")
BEGIN_CARDS = "<!-- BEGIN_CARDS -->"
END_CARDS = "<!-- END_CARDS -->"


def read_csv_execution_queue() -> list[dict[str, str]]:
    return sync_hub.read_rows(CSV_PATH)


def status_to_emoji(status: str) -> str:
    return sync_hub._status_theme(status)["badge"]


def generate_hub_cards(issues: list[dict[str, str]]) -> str:
    return sync_hub.build_cards(issues)


def update_hub_markdown(cards: str) -> str:
    content = HUB_PATH.read_text(encoding="utf-8")
    if BEGIN_CARDS in content and END_CARDS in content:
        return sync_hub.replace_block(content, BEGIN_CARDS, END_CARDS, cards)
    return sync_hub.replace_block(content, sync_hub.CARDS_BEGIN_MARKER, sync_hub.CARDS_END_MARKER, cards)


def main() -> int:
    rows = read_csv_execution_queue()
    cards = generate_hub_cards(rows)
    updated = update_hub_markdown(cards)
    HUB_PATH.write_text(updated, encoding="utf-8")
    print(f"Sincronizado: {HUB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
