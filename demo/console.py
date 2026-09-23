"""Console formatting helpers. Purely cosmetic - no pattern logic lives here."""

from __future__ import annotations

import shutil
import textwrap

WIDTH = min(shutil.get_terminal_size((88, 24)).columns, 88)


def title(text: str, subtitle: str | None = None) -> None:
    print("\n" + "=" * WIDTH)
    print(text)
    if subtitle:
        print(subtitle)
    print("=" * WIDTH)


def phase(text: str) -> None:
    dashes = max(3, WIDTH - len(text) - 4)
    print(f"\n-- {text} " + "-" * dashes)


def call(agent: str, note: str = "") -> None:
    print(f"\n  [{agent}]" + (f"  ({note})" if note else ""))


def body(text: str, indent: int = 6) -> None:
    """Print text with each source line wrapped, preserving line breaks."""
    pad = " " * indent
    for raw_line in str(text).splitlines() or [""]:
        if not raw_line.strip():
            print()
            continue
        for line in textwrap.wrap(raw_line, max(20, WIDTH - indent)):
            print(pad + line)


def para(text: str, indent: int = 6) -> None:
    """Print prose, re-flowing each blank-line-separated paragraph.

    Use this for model output that was hard-wrapped at some other width.
    """
    pad = " " * indent
    blocks = [b for b in str(text).split("\n\n")]
    for i, block in enumerate(blocks):
        flowed = " ".join(block.split())
        if not flowed:
            continue
        if i:
            print()
        for line in textwrap.wrap(flowed, max(20, WIDTH - indent)):
            print(pad + line)


def kv(label: str, value: object, indent: int = 6, pad: int = 16) -> None:
    prefix = " " * indent + f"{label:<{pad}} "
    lines = textwrap.wrap(str(value), max(20, WIDTH - len(prefix))) or [""]
    print(prefix + lines[0])
    for line in lines[1:]:
        print(" " * len(prefix) + line)


def note(text: str, indent: int = 2) -> None:
    body(text, indent=indent)


def result(text: str) -> None:
    print("\n" + "=" * WIDTH)
    body(text, indent=2)
    print("=" * WIDTH)
