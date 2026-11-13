"""The one fence parser for the book's MyST pages, shared by the tools and the tests.

A fence line opens or closes a block: three or more backticks, tildes or colons, then an info string. A fence
is code unless it opens a MyST directive whose content is prose: an info string starting with `{name}` where
name is not in CODE_DIRECTIVES ({admonition}, {note}, {figure}, {table}, ...), or a colon fence with no
directive. A code directive is code behind either fence (:::{code-cell} too). The content
of a prose directive is prose, and the code fences nested in it are code again. A closing fence has no info
string, uses the opening fence's character and is at least as long; so a longer outer fence (four backticks)
protects the fences it contains.

line_kinds(text)   one kind per line: PROSE, DIRECTIVE (the opening or closing fence of a prose directive, or
                   prose inside one) or CODE (a code fence's lines, fences included)
blank(text, kinds, keep_offsets)   the text with the lines of the given kinds emptied (or, with keep_offsets,
                   replaced by spaces, so character offsets are kept too); line numbers are always kept
blank_code(text)   blank(text, {CODE}, keep_offsets=True), for finding math
fences(text)       the code fences as dicts {info, body, line, end}
python_fences(text)  the fences whose info string is exactly `python`
"""
from __future__ import annotations

import re

# MyST directives whose content is code or markup, not prose.
CODE_DIRECTIVES = {"code-cell", "code-block", "sourcecode", "code", "math", "literalinclude", "raw", "eval-rst",
                   "mermaid", "graphviz"}
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,}|:{3,})[ \t]*(\{[^}\n]*\})?(.*?)[ \t]*$")

PROSE, FENCE_LINE, DIRECTIVE, CODE = "prose", "fence", "directive", "code"


def directive_name(role: str | None) -> str:
    """'admonition' for '{admonition}'; '' for no role."""
    words = (role or "")[1:-1].split()
    return words[0] if words else ""


def closes(line: str, code: dict) -> bool:
    """True when `line` closes the open code fence `code`."""
    m = FENCE.match(line)
    return bool(m) and not m.group(2) and not m.group(3) and m.group(1)[0] == code["char"] \
        and len(m.group(1)) >= code["size"]


def _walk(text: str):
    """(line, kind, code) per line, kind in PROSE, FENCE_LINE (a prose directive's own fence), DIRECTIVE (prose
    inside a directive) or CODE; `code` is the open code fence as {char, size, info, line} on its lines."""
    stack, code = [], None   # stack of (char, size) for the open prose directives
    for number, line in enumerate(text.split("\n"), start=1):
        m = FENCE.match(line)
        if code:
            yield line, CODE, code
            if closes(line, code):
                code = None
            continue
        if m:
            char, size, role, rest = m.group(1)[0], len(m.group(1)), m.group(2), m.group(3)
            if not role and not rest and stack and stack[-1][0] == char and size >= stack[-1][1]:
                stack.pop()
                yield line, FENCE_LINE, None
                continue
            if (role and directive_name(role) not in CODE_DIRECTIVES) or (char == ":" and not role):
                stack.append((char, size))
                yield line, FENCE_LINE, None
                continue
            code = {"char": char, "size": size, "info": ((role or "") + rest).strip(), "line": number}
            yield line, CODE, code
            continue
        yield line, (DIRECTIVE if stack else PROSE), None


def line_kinds(text: str) -> list[str]:
    """One of PROSE, FENCE_LINE, DIRECTIVE or CODE per line of `text`."""
    return [kind for _line, kind, _code in _walk(text)]


def blank(text: str, kinds, keep_offsets: bool = False) -> str:
    """The text with every line whose kind is in `kinds` emptied, or replaced by spaces with `keep_offsets`."""
    return "\n".join((" " * len(line) if keep_offsets else "") if kind in kinds else line
                     for line, kind, _code in _walk(text))


def blank_code(text: str) -> str:
    """The text with every line of a code fence replaced by spaces: offsets and line numbers are kept, and the
    prose and math of directives (admonitions, figure captions, tables) stay readable."""
    return blank(text, {CODE}, keep_offsets=True)


def fences(text: str) -> list[dict]:
    """The code fences of `text` as dicts {info, body, line, end}: info string, body, 1-based line of the opening
    fence, and the character offset at the end of the closing fence line. Fences inside a prose directive are
    found; directive fences are not returned. An unclosed fence is not returned."""
    found, offset, current, body_start = [], 0, None, 0
    for line, kind, code in _walk(text):
        end = offset + len(line)
        if kind == CODE and code is not current:
            current, body_start = code, end + 1
        elif kind == CODE and closes(line, code):
            found.append({"info": code["info"], "body": text[body_start:offset], "line": code["line"], "end": end})
            current = None
        offset = end + 1
    return found


def python_fences(text: str) -> list[dict]:
    """The fences of `text` whose info string is exactly `python` (the runnable listings), as from fences()."""
    return [f for f in fences(text) if f["info"] == "python"]
