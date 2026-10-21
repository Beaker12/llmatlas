#!/usr/bin/env python3
"""Check that every arXiv id in references.bib resolves to the paper the entry names.

For each entry with an ``eprint`` field the title in the entry is compared with the title the
arXiv API returns for that id (case, punctuation and TeX braces ignored; the first 30
characters of the entry's title must be a prefix of the arXiv title). Exits non-zero on any
mismatch. Standard library only; requests are batched and spaced to respect arXiv's limits.

Usage: python tools/verify_arxiv.py [references.bib] [--keys key1,key2,...]
"""
from __future__ import annotations

import html
import re
import sys
import time
import urllib.parse
import urllib.request

BATCH = 20


def normalise(title: str) -> str:
    title = re.sub(r"[{}\\]", "", title)
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def entries(text):
    for m in re.finditer(r"@\w+\{([^,]+),([\s\S]*?)\n\}", text):
        body = m.group(2)
        eprint = re.search(r"eprint\s*=\s*\{([^}]+)\}", body)
        title = re.search(r"title\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        if eprint and title:
            yield m.group(1), eprint.group(1).strip(), title.group(1)


def fetch_titles(ids):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    for attempt in range(4):
        try:
            xml = urllib.request.urlopen(url, timeout=60).read()
            break
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("arXiv API unreachable")
    # The Atom response is read with regular expressions rather than an XML parser: stdlib
    # parsers expand entities from untrusted input, and defusedxml is not a dependency.
    titles = {}
    for block in xml.decode("utf-8", "replace").split("<entry>")[1:]:
        ident = re.search(r"<id>([^<]+)</id>", block)
        title = re.search(r"<title>([\s\S]*?)</title>", block)
        if ident and title:
            arxiv_id = re.sub(r"v\d+$", "", ident.group(1).rsplit("/abs/", 1)[-1])
            titles[arxiv_id] = " ".join(html.unescape(title.group(1)).split())
    return titles


def main(argv):
    path = "references.bib"
    only = None
    args = list(argv)
    if "--keys" in args:
        i = args.index("--keys")
        only = set(args[i + 1].split(","))
        del args[i : i + 2]
    if args:
        path = args[0]
    items = [(k, e, t) for k, e, t in entries(open(path, encoding="utf-8").read())
             if only is None or k in only]
    bad = 0
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        titles = fetch_titles([e for _, e, _ in batch])
        for key, eprint, title in batch:
            found = titles.get(eprint)
            ok = found is not None and normalise(found).startswith(normalise(title)[:30])
            bad += not ok
            print(("OK  " if ok else "BAD ") + f"{key:28} {eprint:12} {(found or 'NOT FOUND')[:70]}")
        time.sleep(3)
    print(f"{len(items)} entries checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
