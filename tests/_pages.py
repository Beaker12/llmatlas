"""Reading chapter pages for the page-level checks.

pages(chapter_dir)       every *.md directly in the chapter directory, sorted
blank_code(text)         the text with the lines of code fences replaced by empty lines (line numbers kept)
blank_math(text)         the text with display and inline math replaced by spaces (positions kept)
blank_inline_code(text)  the text with inline code spans replaced by spaces (positions kept)
line_of(text, pos)       1-based line number of a character offset

A fence is code unless it opens a MyST directive (info string starting with `{`, other than the code-like
directives below) or is a colon fence; the content of a directive such as an admonition or a figure is prose.
"""
import re
from pathlib import Path

CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "code-cell", "literalinclude", "math", "mermaid", "graphviz"}
FENCE = re.compile(r"^\s*(`{3,}|~{3,}|:{3,})\s*(\{[^}\n]*\})?(.*)$")
CLOSING = re.compile(r"^\s*(`{3,}|~{3,}|:{3,})\s*$")


def pages(chapter_dir):
    return sorted(Path(chapter_dir).glob("*.md"))


def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def _directive_name(role):
    words = role[1:-1].split()
    return words[0] if words else ""


def blank_code(text: str) -> str:
    out, code, stack = [], None, []
    for line in text.split("\n"):
        if code:
            m = CLOSING.match(line)
            if m and m.group(1)[0] == code[0] and len(m.group(1)) >= code[1]:
                code = None
            out.append("")
            continue
        m = FENCE.match(line)
        if not m:
            out.append(line)
            continue
        char, size, role, rest = m.group(1)[0], len(m.group(1)), m.group(2), m.group(3)
        if not role and not rest.strip() and stack and stack[-1][0] == char and size >= stack[-1][1]:
            stack.pop()
            out.append(line)
        elif char == ":" or (role and _directive_name(role) not in CODE_DIRECTIVES):
            if role:
                stack.append((char, size))
            out.append(line)
        else:
            code = (char, size)
            out.append("")
    return "\n".join(out)


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
