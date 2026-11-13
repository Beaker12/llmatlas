"""Reading chapter pages for the page-level checks.

pages(chapter_dir)       every *.md directly in the chapter directory except notation-local.md, sorted
blank_code(text)         the text with the lines of code fences replaced by empty lines (line numbers kept)
blank_math(text)         the text with display and inline math replaced by spaces (positions kept)
blank_inline_code(text)  the text with inline code spans replaced by spaces (positions kept)
line_of(text, pos)       1-based line number of a character offset

fences(text)             the code fences (tools/fences.py, the one fence parser, which also defines what is code)
"""
import re

from tools.book_files import pages  # noqa: F401  (the one page list, shared with the tools)
from tools.fences import CODE, DIRECTIVE, FENCE_LINE, blank, fences  # noqa: F401  (the one fence parser)

def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def blank_code(text: str, directives: bool = False) -> str:
    """Blank the lines of code fences; with `directives`, also those of every directive (admonition, figure...)."""
    return blank(text, {CODE, FENCE_LINE, DIRECTIVE} if directives else {CODE})


def blank_math(text: str) -> str:
    text = re.sub(r"\$\$[\s\S]*?\$\$", _blank, text)
    return re.sub(r"(?<![\\\w])\$[^$\n]+\$", _blank, text)


def blank_inline_code(text: str, keep_roles: bool = False) -> str:
    """Blank inline code spans; with keep_roles, leave the target of a role such as {cite}`key` in place."""
    if keep_roles:
        return re.sub(r"(\{[\w:-]+\}`[^`\n]*`)|``[^\n]*?``|`[^`\n]*`",
                      lambda m: m.group(0) if m.group(1) else _blank(m), text)
    return re.sub(r"``[^\n]*?``|`[^`\n]*`", _blank, text)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1
