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
import urllib.error
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


PARTICLES = {"van", "von", "de", "der", "den", "la", "le", "di", "du", "del", "da", "dos",
             "ter", "ten", "al", "el", "bin"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def name_words(name: str, keep_particles: bool = False):
    """Lower-case ASCII words of a name without generational suffixes and (unless asked to keep
    them) particles such as ``van``; a name made only of particles keeps them."""
    words = [w for w in re.findall(r"[a-z0-9]+", normalise_words(name)) if w not in SUFFIXES]
    if keep_particles:
        return words
    return [w for w in words if w not in PARTICLES] or words


def normalise_words(text: str) -> str:
    text = re.sub(r"\\['`\"^~=.vcHuk]\s*\{?\\?([A-Za-z])\}?", r"\1", text)  # TeX accents
    return fold(re.sub(r"[{}\\]", "", text)).lower().replace("-", "")


def bib_surnames(author_field: str):
    """Surname keys of the entry's authors: ``None`` for a name that folds to nothing (a
    non-Latin script), ``"others"`` for ``and others``, and ``("corporate", key)`` for a name
    wrapped in braces."""
    names = [a.strip() for a in re.split(r"\s+and\s+", author_field) if a.strip()]
    surnames = []
    for name in names:
        if name == "others":
            surnames.append("others")
        elif name.startswith("{") and name.endswith("}"):  # corporate author
            key = "".join(name_words(name))
            surnames.append(("corporate", key) if key else None)
        else:
            part = name.split(",")[0] if "," in name else name
            words = name_words(part if "," in name else part.split()[-1] if part.split() else "")
            surnames.append("".join(words) or None)
    return surnames


def record_name_keys(name: str):
    """Acceptable surname keys for a record author written ``Given Family``: the last word, or
    the last two or three words joined (compound surnames)."""
    keys = set()
    for words in (name_words(name), name_words(name, keep_particles=True)):
        keys |= {"".join(words[-k:]) for k in (1, 2, 3) if len(words) >= k}
    return keys


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


# A trailing event suffix on a Crossref container title, as in "... Theory of Computing - STOC '98".
EVENT_SUFFIX = re.compile(r"\s+-\s+[A-Z][\w-]*\s*'\d{2}\s*$")


def crossref_author(author) -> str:
    """A Crossref author as written, given names then family name, so a compound surname that Crossref splits
    between the two (given "Loubna Ben", family "Allal") still yields its whole surname; else the name field."""
    name = " ".join(part for part in (author.get("given"), author.get("family")) if part)
    return name or author.get("name", "")


def parse_crossref(payload: str):
    msg = json.loads(payload)["message"]
    container = EVENT_SUFFIX.sub("", (msg.get("container-title") or [""])[0])
    return {
        "title": (msg.get("title") or [""])[0],
        "authors": [crossref_author(a) for a in msg.get("author", [])],
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
    """Problems (a list of strings) comparing the entry's authors with a record's. Surnames are
    compared whole, after folding case, accents, braces, hyphens and particles."""
    mine = bib_surnames(bib_author)
    open_ended = bool(mine) and mine[-1] == "others"
    if open_ended:
        mine = mine[:-1]
    record_authors = [a for a in record_authors if name_words(a) or re.search(r"\w", a)]
    if not record_authors:
        return []
    if len(mine) != len(record_authors) and not (open_ended and len(mine) <= len(record_authors)):
        return [f"author count {len(mine)} in the entry, {len(record_authors)} in the record"]
    problems = []
    for i, name in enumerate(mine):
        theirs = record_name_keys(record_authors[i])
        if name is None or not theirs:
            problems.append(f"could not compare author {i + 1} ({record_authors[i]!r}): the name "
                            "has no Latin letters; check it by hand")
        elif isinstance(name, tuple):
            if not any(name[1] in t or t in name[1] for t in theirs):
                problems.append(f"author {i + 1} is {name[1]!r} in the entry, "
                                f"{record_authors[i]!r} in the record")
        elif name not in theirs:
            problems.append(f"author {i + 1} is {name!r} in the entry, "
                            f"{record_authors[i]!r} in the record")
        if problems:
            return problems[:1]
    return []


def is_preprint_venue(venue: str) -> bool:
    return "arxiv" in venue.lower()


def venue_of(fields):
    return fields.get("journal") or fields.get("booktitle") or ""


def has_real_venue(fields) -> bool:
    """True for an entry published in a journal or proceedings; an arXiv preprint string in
    ``journal`` does not count."""
    venue = venue_of(fields)
    return bool(venue) and not is_preprint_venue(venue)


STOP_WORDS = {"of", "the", "on", "and", "in", "for"}
BOILERPLATE = {"proceedings", "proc", "conference", "international", "annual", "meeting",
               "workshop", "symposium", "ieee", "acm", "published", "at"}


def venue_tokens(venue: str):
    """Word tokens of a venue with parenthesised text (``(CVPR)``), years and ordinals removed
    and stop words dropped; case is kept."""
    text = re.sub(r"\([^)]*\)", " ", normalise_words_keep_case(venue))
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    return [t for t in tokens if t.lower() not in STOP_WORDS and not re.fullmatch(
        r"\d+(st|nd|rd|th)?", t.lower())]


def normalise_words_keep_case(text: str) -> str:
    text = re.sub(r"\\['`\"^~=.vcHuk]\s*\{?\\?([A-Za-z])\}?", r"\1", text)  # TeX accents
    return fold(re.sub(r"[{}\\]", "", text)).replace("-", "")


def significant(venue: str):
    """Tokens without boilerplate such as ``Proceedings`` and ``Conference``."""
    return [t for t in venue_tokens(venue) if t.lower() not in BOILERPLATE]


def word_match(x: str, y: str) -> bool:
    """Equal, or one is a prefix abbreviation of the other (``nat`` and ``nature``), ignoring
    case. ``Neural Comput.`` abbreviates both ``Neural Computation`` and ``Neural Computing``;
    that ambiguity is inherent in abbreviations and is left as a match."""
    x, y = x.lower(), y.lower()
    return x.startswith(y) or y.startswith(x)


def capitals(token: str) -> str:
    return "".join(c for c in token if c.isupper())


def is_initialism(token: str) -> bool:
    """A token of at most eight letters with two or more capitals: ``ACL``, ``NeurIPS``."""
    return len(token) <= 8 and len(capitals(token)) >= 2 and token.isalpha()


def spells(token: str, venue: str) -> bool:
    """The token's capitals are the initials of a run of consecutive words of the venue (stop
    words skipped, boilerplate counted: ``ICML`` is International Conference on Machine
    Learning)."""
    initials = "".join(t[0].lower() for t in venue_tokens(venue))
    return capitals(token).lower() in initials


def same_venue(mine: str, theirs: str) -> bool:
    """Do two venue names denote the same venue? Boilerplate, stop words, years and ordinals are
    ignored on both sides. Otherwise the two must have the same number of significant words and
    each pair must match (equal, or one a prefix abbreviation of the other), so ``Nature`` does
    not match ``Nature Communications``. The exception is a lone initialism (``ACL``, ``JMLR``,
    ``NeurIPS``), which matches a venue whose words' initials spell it. A venue with no
    significant words matches nothing but another such venue."""
    a, b = significant(mine), significant(theirs)
    if not a or not b:
        return not a and not b
    if len(a) == 1 and len(b) == 1 and a[0].lower() == b[0].lower():
        return True
    if len(a) == 1 and is_initialism(a[0]):
        return spells(a[0], theirs)
    if len(b) == 1 and is_initialism(b[0]):
        return spells(b[0], mine)
    return len(a) == len(b) and all(word_match(x, y) for x, y in zip(a, b))


def norm_pages(pages: str) -> str:
    return re.sub(r"[-–—]+", "-", pages.replace(" ", ""))


def compare_publication(fields, journal, volume, pages, source):
    problems = []
    mine = venue_of(fields)
    # A reference cut at its first digit ("Proceedings of the 57th ...") may keep only boilerplate;
    # that says nothing about the venue, so it is not compared.
    if (journal and significant(journal) and mine and not is_preprint_venue(mine)
            and not same_venue(mine, journal)):
        problems.append(f"journal: {mine!r} in the entry, {journal!r} in {source}")
    if volume and fields.get("volume") and fields["volume"].strip() != str(volume).strip():
        problems.append(f"volume: {fields['volume']} in the entry, {volume} in {source}")
    if pages and fields.get("pages") and norm_pages(fields["pages"]) != norm_pages(pages):
        problems.append(f"pages: {fields['pages']} in the entry, {pages} in {source}")
    return problems


def compare_arxiv(fields, record):
    """Problems for one entry against its arXiv record. The title is compared for entries not
    published in a real venue (preprints, including a ``journal`` that is an arXiv preprint
    string); a published title may differ from the preprint's."""
    if record is None:
        return ["arXiv has no record for this id"]
    problems = []
    if fields.get("author"):
        problems += authors_match(fields["author"], record["authors"])
    if not has_real_venue(fields) and fields.get("title"):
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


class NotFound(Exception):
    """The service answered that it has no such record."""


class Unreachable(Exception):
    """The service could not be reached or answered with an error."""


def http_get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise NotFound(url) from error
            time.sleep(5 * (attempt + 1))
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise Unreachable(url)


class ArxivError(Exception):
    """arXiv answered, but not with a usable result; the message says what to report."""


def fetch_arxiv(ids):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    try:
        text = http_get(url)
    except NotFound as error:
        raise ArxivError("arXiv endpoint returned 404") from error
    records = parse_atom(text)
    if "<feed" not in text or not records:
        raise ArxivError("could not read the arXiv response")
    return records


def fetch_crossref(doi: str):
    """The Crossref record, ``None`` when Crossref has none; ``Unreachable`` on network failure."""
    try:
        return parse_crossref(http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))
    except NotFound:
        return None
    except (ValueError, KeyError) as error:
        raise Unreachable("unreadable Crossref response") from error


def check_entry(fields, arxiv_record, crossref=fetch_crossref, delay=CROSSREF_DELAY):
    """All problems for one entry. A DOI the entry lacks but arXiv supplies is checked too."""
    problems = []
    if "eprint" in fields:
        problems += compare_arxiv(fields, arxiv_record)
    doi = fields.get("doi") or (arxiv_record or {}).get("doi")
    if doi:
        try:
            record = crossref(doi.strip())
            time.sleep(delay)
        except Unreachable:
            return problems + ["could not reach Crossref"]
        problems += ["Crossref has no record for this DOI"] if record is None \
            else compare_crossref(fields, record)
    return problems


def select_entries(paths, only=None):
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            items += [(k, f) for k, f in parse_entries(handle.read())
                      if ("eprint" in f or "doi" in f) and (only is None or k in only)]
    return items


def main(argv):
    only = None
    args = list(argv)
    if "--keys" in args:
        i = args.index("--keys")
        only = set(args[i + 1].split(","))
        del args[i:i + 2]
    paths = args or ["references.bib"] + sorted(glob.glob("bib/*.bib"))
    items = select_entries(paths, only)
    if not items:
        what = f"matches --keys {','.join(sorted(only))}" if only else "has an eprint or doi field"
        print(f"nothing checked: no entry in {', '.join(paths)} {what}")
        return 2
    if only:
        missing = only - {k for k, _ in items}
        if missing:
            print("no entry with an eprint or doi for: " + ", ".join(sorted(missing)))
    bad = 0
    arxiv_items = [(k, f) for k, f in items if "eprint" in f]
    records = {}
    try:
        for start in range(0, len(arxiv_items), BATCH):
            batch = arxiv_items[start:start + BATCH]
            records.update(fetch_arxiv([f["eprint"].strip() for _, f in batch]))
            time.sleep(DELAY)
    except Unreachable:
        print("could not reach arXiv")
        return 2
    except ArxivError as error:
        print(str(error))
        return 2
    for key, fields in items:
        problems = check_entry(fields, records.get(fields.get("eprint", "").strip()))
        bad += bool(problems)
        ident = fields.get("eprint", fields.get("doi", ""))
        print(("OK  " if not problems else "BAD ") + f"{key:28} {ident}")
        for problem in problems:
            print("      " + problem)
    print(f"{len(items)} entries checked, {bad} mismatched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
