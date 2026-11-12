#!/usr/bin/env python3
"""Check that bibliography entries match the arXiv and Crossref records they point to.

For each entry with an ``eprint`` field the arXiv API record is compared with the entry:

* authors: the list of surnames must be the same, in the same order (``and others`` in the
  entry truncates the comparison to the names given);
* title: compared (case, punctuation and TeX braces ignored; the first 30 characters of the
  entry's title must be a prefix of the arXiv title) only for entries without a ``journal`` or
  ``booktitle``, because a published title may differ from the preprint's;
* when arXiv supplies ``journal_ref``: the entry's ``journal`` or ``booktitle``, ``volume`` and
  ``pages`` are compared with it (a field is skipped when either side lacks it).

An entry with a ``doi`` field is also compared with its Crossref record on authors,
``journal``/``booktitle``, ``volume`` and ``pages``. Entries whose journal is an arXiv preprint
are not compared on journal. Exits non-zero on any mismatch. Standard library only; requests are
batched and spaced to respect the services' limits.

Usage: python tools/verify_arxiv.py [file.bib ...] [--keys key1,key2,...]
With no file the tool reads references.bib and every bib/*.bib.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

FIELD = (r"(\w+)\s*=\s*(\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
         r"|\"([^\"]*)\"|(\w+))")
BATCH = 20
DELAY = 3.0
CROSSREF_DELAY = 1.0
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; bib-checker)")


# --- parsing -------------------------------------------------------------------------------


def normalise(title: str) -> str:
    title = re.sub(r"\\['`\"^~=.vcHuk]\s*\{?\\?([A-Za-z])\}?", r"\1", title)  # TeX accents
    title = re.sub(r"[{}\\]", "", title)
    return re.sub(r"[^a-z0-9]+", "", fold(title).lower())


def fold(text: str) -> str:
    """ASCII-fold accents so that "Müller" and "Muller" compare equal."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def parse_entries(text):
    """Yield ``(key, fields)`` for each entry; field values lose one level of braces or quotes."""
    for m in re.finditer(r"@\w+\{([^,]+),([\s\S]*?)\n\}", text):
        fields = {}
        for f in re.finditer(FIELD, m.group(2)):
            value = f.group(3) if f.group(3) is not None else (f.group(4) or f.group(5))
            fields[f.group(1).lower()] = value.strip()
        yield m.group(1).strip(), fields


def entries(text):
    """``(key, eprint, title)`` for each entry with both fields (the original interface)."""
    for key, fields in parse_entries(text):
        if "eprint" in fields and "title" in fields:
            yield key, fields["eprint"], fields["title"]


def bib_surnames(author_field: str):
    names = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
    surnames = []
    for name in names:
        if name == "others":
            surnames.append("others")
        elif name.startswith("{") and name.endswith("}"):  # corporate author
            surnames.append(normalise(name))
        elif "," in name:
            surnames.append(normalise(name.split(",")[0]))
        else:
            surnames.append(normalise(name.split()[-1]))
    return surnames


def full_name_key(name: str) -> str:
    return normalise(name)


# --- records from the services ----------------------------------------------------------------


def parse_atom(xml: str):
    """Map arXiv id -> record from an Atom response. Regular expressions are used rather than
    an XML parser: stdlib parsers expand entities from untrusted input and defusedxml is not a
    dependency."""
    records = {}
    for block in xml.split("<entry>")[1:]:
        ident = re.search(r"<id>([^<]+)</id>", block)
        title = re.search(r"<title>([\s\S]*?)</title>", block)
        if not (ident and title):
            continue
        arxiv_id = re.sub(r"v\d+$", "", ident.group(1).rsplit("/abs/", 1)[-1])

        def tag(name):
            m = re.search(rf"<arxiv:{name}[^>]*>([\s\S]*?)</arxiv:{name}>", block)
            return " ".join(html.unescape(m.group(1)).split()) if m else None

        authors = [" ".join(html.unescape(a).split())
                   for a in re.findall(r"<author>\s*<name>([\s\S]*?)</name>", block)]
        records[arxiv_id] = {
            "title": " ".join(html.unescape(title.group(1)).split()),
            "authors": authors,
            "journal_ref": tag("journal_ref"),
            "doi": tag("doi"),
        }
    return records


def parse_crossref(payload: str):
    msg = json.loads(payload)["message"]
    container = (msg.get("container-title") or [""])[0]
    return {
        "title": (msg.get("title") or [""])[0],
        "authors": [a.get("family") or a.get("name", "") for a in msg.get("author", [])],
        "journal": html.unescape(container),
        "volume": msg.get("volume"),
        "pages": msg.get("page"),
    }


def split_journal_ref(ref: str):
    """Best-effort ``(journal, volume, pages)`` from a free-text journal reference."""
    m = re.search(r"\d", ref)
    journal = ref[: m.start()] if m else ref
    rest = ref[m.start():] if m else ""
    first = re.match(r"\d+", rest)
    # A leading year (as in "Nature 2020") is not a volume.
    volume = first.group(0) if first and not re.fullmatch(r"(19|20)\d\d", first.group(0)) else None
    pages = re.search(r"(?<![\d(])(\d+)\s*[-–—]+\s*(\d+)", rest)
    return journal.strip(" ,;:."), volume, (f"{pages.group(1)}-{pages.group(2)}" if pages else None)


# --- comparison ---------------------------------------------------------------------------------


def authors_match(bib_author: str, record_authors):
    """Problems (a list of strings) comparing the entry's authors with a record's."""
    mine = bib_surnames(bib_author)
    open_ended = bool(mine) and mine[-1] == "others"
    if open_ended:
        mine = mine[:-1]
    theirs = [full_name_key(a) for a in record_authors]
    if not theirs:
        return []
    if len(mine) != len(theirs) and not (open_ended and len(mine) <= len(theirs)):
        return [f"author count {len(mine)} in the entry, {len(theirs)} in the record"]
    for i, name in enumerate(mine):
        last = normalise(record_authors[i].split()[-1]) if record_authors[i].split() else ""
        if not (theirs[i].endswith(name) or (last and name.startswith(last))):
            return [f"author {i + 1} is {name!r} in the entry, {record_authors[i]!r} in the record"]
    return []


def is_preprint_venue(venue: str) -> bool:
    return "arxiv" in venue.lower()


def venue_of(fields):
    return fields.get("journal") or fields.get("booktitle") or ""


GENERIC_WORDS = {"proceedings", "of", "the", "on", "and", "in", "conference", "international",
                 "annual", "journal", "ieee", "acm", "workshop", "published", "at", "for",
                 "meeting"}


def venue_words(venue: str):
    words = re.findall(r"[a-z]+", fold(re.sub(r"[{}\\]", "", venue)).lower())
    return {w for w in words if w not in GENERIC_WORDS}


def same_venue(mine: str, theirs: str) -> bool:
    """True when one venue's significant words are contained in the other's. Undecidable
    comparisons (a side with no significant words, or one bare acronym) count as a match."""
    a, b = venue_words(mine), venue_words(theirs)
    if not a or not b or min(len(a), len(b)) < 2 and a != b and (len(a) == 1 or len(b) == 1):
        return True
    return a <= b or b <= a


def norm_pages(pages: str) -> str:
    return re.sub(r"[-–—]+", "-", pages.replace(" ", ""))


def compare_publication(fields, journal, volume, pages, source):
    problems = []
    mine = venue_of(fields)
    if journal and mine and not is_preprint_venue(mine) and not same_venue(mine, journal):
        problems.append(f"journal: {mine!r} in the entry, {journal!r} in {source}")
    if volume and fields.get("volume") and fields["volume"].strip() != str(volume).strip():
        problems.append(f"volume: {fields['volume']} in the entry, {volume} in {source}")
    if pages and fields.get("pages") and norm_pages(fields["pages"]) != norm_pages(pages):
        problems.append(f"pages: {fields['pages']} in the entry, {pages} in {source}")
    return problems


def compare_arxiv(fields, record):
    """Problems for one entry against its arXiv record."""
    if record is None:
        return ["arXiv has no record for this id"]
    problems = []
    if fields.get("author"):
        problems += authors_match(fields["author"], record["authors"])
    if not venue_of(fields) and fields.get("title"):
        if not normalise(record["title"]).startswith(normalise(fields["title"])[:30]):
            problems.append(f"title: arXiv has {record['title'][:70]!r}")
    if record.get("journal_ref") and not is_preprint_venue(venue_of(fields)):
        journal, volume, pages = split_journal_ref(record["journal_ref"])
        problems += compare_publication(fields, journal, volume, pages,
                                        "the arXiv journal reference")
    return problems


def compare_crossref(fields, record):
    problems = []
    if fields.get("author"):
        problems += authors_match(fields["author"], record["authors"])
    problems += compare_publication(fields, record["journal"], record["volume"], record["pages"],
                                    "Crossref")
    return problems


# --- network ------------------------------------------------------------------------------------


def http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"unreachable: {url}")


def fetch_arxiv(ids):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    return parse_atom(http_get(url))


def fetch_crossref(doi: str):
    try:
        return parse_crossref(http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))
    except (RuntimeError, ValueError, KeyError):
        return None


def main(argv):
    paths, only = [], None
    args = list(argv)
    if "--keys" in args:
        i = args.index("--keys")
        only = set(args[i + 1].split(","))
        del args[i:i + 2]
    paths = args or ["references.bib"] + sorted(glob.glob("bib/*.bib"))
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            items += [(k, f) for k, f in parse_entries(handle.read())
                      if ("eprint" in f or "doi" in f) and (only is None or k in only)]
    bad = 0
    arxiv_items = [(k, f) for k, f in items if "eprint" in f]
    records = {}
    for start in range(0, len(arxiv_items), BATCH):
        batch = arxiv_items[start:start + BATCH]
        records.update(fetch_arxiv([f["eprint"].strip() for _, f in batch]))
        time.sleep(DELAY)
    for key, fields in items:
        problems = []
        if "eprint" in fields:
            problems += compare_arxiv(fields, records.get(fields["eprint"].strip()))
        if "doi" in fields:
            record = fetch_crossref(fields["doi"].strip())
            time.sleep(CROSSREF_DELAY)
            problems += ["Crossref has no record for this DOI"] if record is None \
                else compare_crossref(fields, record)
        bad += bool(problems)
        ident = fields.get("eprint", fields.get("doi", ""))
        print(("OK  " if not problems else "BAD ") + f"{key:28} {ident}")
        for problem in problems:
            print("      " + problem)
    print(f"{len(items)} entries checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
