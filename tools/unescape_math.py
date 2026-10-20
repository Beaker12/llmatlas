#!/usr/bin/env python3
"""Repair mdformat-style escaping inside MyST math regions.

mdformat (without the MyST plugin) rewrote ``\\frac`` as ``\\\\frac``, ``_`` as ``\\_``,
``[``/``]`` as ``\\[``/``\\]``, ``<`` as ``\\<`` and ``*`` as ``\\*`` inside ``$...$``
and ``$$...$$`` regions.  This module finds math regions while skipping fenced code
blocks, undoes those escapes only inside math, and leaves a trailing LaTeX row
separator (``\\\\`` followed by whitespace or end of line) untouched.

Usage:
    python tools/unescape_math.py --check FILE...      # dry run, print counts
    python tools/unescape_math.py --write FILE...      # rewrite in place
    python tools/unescape_math.py --write --expect 52 FILE   # fail unless N lines changed
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE = re.compile(r"^(```|~~~)[^\n]*\n[\s\S]*?^\1[ \t]*$", re.MULTILINE)
DISPLAY = re.compile(r"\$\$[\s\S]*?\$\$")
INLINE = re.compile(r"(?<![\\$])\$(?!\$)[^$\n]+?(?<!\\)\$(?!\$)")

# Order matters: the doubled-backslash rule runs first on the original text.
RULES = [
    (re.compile(r"\\\\(?=[A-Za-z{(])"), "\\"),
    (re.compile(r"\\_"), "_"),
    (re.compile(r"\\\["), "["),
    (re.compile(r"\\\]"), "]"),
    (re.compile(r"\\<"), "<"),
    (re.compile(r"\\\*"), "*"),
]
LEFTOVER = re.compile(r"\\\\(?!\s|$)")


def _blank_out(text: str, pattern: re.Pattern) -> str:
    """Replace every match with spaces/newlines so offsets are preserved."""

    def repl(m: re.Match) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in m.group(0))

    return pattern.sub(repl, text)


def math_regions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of every math region, fenced code excluded."""
    no_code = _blank_out(text, FENCE)
    regions = [(m.start(), m.end()) for m in DISPLAY.finditer(no_code)]
    no_display = _blank_out(no_code, DISPLAY)
    regions += [(m.start(), m.end()) for m in INLINE.finditer(no_display)]
    return sorted(regions)


def unescape_region(region: str) -> str:
    for pattern, replacement in RULES:
        region = pattern.sub(lambda _m, r=replacement: r, region)
    return region


def unescape_text(text: str) -> tuple[str, int, list[str]]:
    """Return (new_text, changed_line_count, leftover_warnings)."""
    out = []
    pos = 0
    changed_lines: set[int] = set()
    leftovers: list[str] = []
    for start, end in math_regions(text):
        out.append(text[pos:start])
        original = text[start:end]
        fixed = unescape_region(original)
        if fixed != original:
            base_line = text[:start].count("\n") + 1
            for i, (a, b) in enumerate(zip(original.split("\n"), fixed.split("\n"))):
                if a != b:
                    changed_lines.add(base_line + i)
        for m in LEFTOVER.finditer(fixed):
            line = text[:start].count("\n") + 1 + fixed[: m.start()].count("\n")
            leftovers.append(f"line {line}: {fixed[max(0, m.start()-10):m.end()+10]!r}")
        out.append(fixed)
        pos = end
    out.append(text[pos:])
    return "".join(out), len(changed_lines), leftovers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="dry run")
    mode.add_argument("--write", action="store_true", help="rewrite files in place")
    ap.add_argument("--expect", type=int, default=None,
                    help="fail unless exactly this many lines change (single file)")
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args(argv)

    rc = 0
    for path in args.files:
        text = path.read_text(encoding="utf-8")
        new_text, n_changed, leftovers = unescape_text(text)
        print(f"{path}: {n_changed} line(s) would change" if args.check
              else f"{path}: {n_changed} line(s) changed")
        for w in leftovers:
            print(f"  LEFTOVER doubled backslash {w}")
            rc = 1
        if args.expect is not None and n_changed != args.expect:
            print(f"  EXPECTED {args.expect} changed lines, got {n_changed}")
            rc = 1
        if args.write and new_text != text:
            path.write_text(new_text, encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
