"""Guards the form of the bibliography: references.bib and bib/*.bib.

Checks (all over the parsed entries; pybtex does the parsing):
- no key defined twice, within a file or across files;
- one paper, one key: no two entries share an eprint, a DOI or a URL (a doi.org URL counts as a DOI,
  an arxiv.org/abs URL as an eprint);
- no raw `&` and no raw em dash in a title, booktitle, journal, publisher, institution or note (write `\\&` and
  `---`);
- every key cited with a `{cite}`, `{cite:t}`, `{cite:p}`... role in a tracked .md file or in a notebook
  markdown cell exists;
- required fields per entry type (see REQUIRED below);
- every entry has an eprint, a doi or a url.

How `@article` entries for preprints are treated: an `@article` whose `journal` starts with "arXiv" and that
has an `eprint` is the preprint form and is exempt from `volume` and `pages`. A journal that has no
volume or page numbers (TMLR) is listed in UNPAGINATED_JOURNALS.
"""
import json
import re

import pytest

from tools.book_files import bib_files as book_bib_files
from pybtex.database import BibliographyData, parse_file
from pybtex.errors import set_strict_mode
from pybtex.database import BibliographyDataError

from tests._files import ROOT, tracked

# Existing entries that miss a required field and whose missing detail could not be verified from a
# fetched source: key -> the fields that are allowed to be absent. Keep this list short; remove a key
# when the entry is completed.
# "identifier" stands for the eprint/doi/url requirement.
KNOWN_INCOMPLETE = {
    # No URL: the MIT Press page refused the automated request (Access Denied) and the open catalogues are
    # not on the permitted source list, so a URL could not be verified. Title, author, publisher, year and
    # ISBN are unchanged.
    "murphy2012machine": {"identifier"},
}

UNPAGINATED_JOURNALS = {"Transactions on Machine Learning Research"}

CITE = re.compile(r"\{cite(?::[a-z]+)?\}`([^`\n]+)`")


def bib_files():
    return book_bib_files(ROOT)


def load(files):
    """[(file name, key, entry)] for every entry; duplicate keys inside one file are reported by pybtex."""
    set_strict_mode(True)
    out, failures = [], []
    for path in files:
        try:
            data = parse_file(str(path), bib_format="bibtex")
        except BibliographyDataError as err:
            failures.append(f"{path.name}: {err}")
            continue
        out.extend((path.name, key, entry) for key, entry in data.entries.items())
    return out, failures


def field(entry, name):
    return entry.fields.get(name, "").strip()


def has_people(entry, role):
    return bool(entry.persons.get(role)) or bool(field(entry, role))


def identifiers(entry):
    """The (kind, value) identifiers of an entry, normalised."""
    ids = set()
    eprint = field(entry, "eprint")
    if eprint:
        ids.add(("eprint", re.sub(r"v\d+$", "", eprint.lower())))
    doi = field(entry, "doi")
    if doi:
        ids.add(("doi", re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.lower().strip())))
    url = field(entry, "url")
    if url:
        low = url.lower().rstrip("/")
        m = re.match(r"https?://(?:dx\.)?doi\.org/(.+)$", low)
        a = re.match(r"https?://arxiv\.org/(?:abs|pdf)/(.+?)(?:v\d+)?(?:\.pdf)?$", low)
        if m:
            ids.add(("doi", m.group(1)))
        elif a:
            ids.add(("eprint", a.group(1)))
        else:
            ids.add(("url", low))
    return ids


def missing_fields(entry):
    kind = entry.type.lower()
    missing = []

    def need(*names):
        missing.extend(n for n in names if not (has_people(entry, n) if n in ("author", "editor") else field(entry, n)))

    if kind == "article":
        need("author", "title", "journal", "year")
        preprint = field(entry, "journal").lower().startswith("arxiv") and field(entry, "eprint")
        if not preprint and field(entry, "journal") not in UNPAGINATED_JOURNALS:
            need("volume", "pages")
    elif kind == "inproceedings":
        need("author", "title", "booktitle", "year")
    elif kind == "book":
        need("title", "publisher", "year")
        if not (has_people(entry, "author") or has_people(entry, "editor")):
            missing.append("author or editor")
    elif kind in ("misc", "techreport"):
        need("title", "year", "url")
    return missing


def gaps(entry):
    """Missing required fields, plus "identifier" when there is no eprint, doi or url."""
    return missing_fields(entry) + ([] if identifiers(entry) else ["identifier"])


TEXT_FIELDS = ("title", "booktitle", "journal", "publisher", "institution", "note")


def check_bibliography(files):
    """Every failure string for the given bib files."""
    entries, failures = load(files)
    seen = {}
    for name, key, entry in entries:
        if key in seen and seen[key] != name:
            failures.append(f"key {key} defined in {seen[key]} and {name}")
        seen.setdefault(key, name)

    owners = {}
    for name, key, entry in entries:
        for ident in identifiers(entry):
            owners.setdefault(ident, []).append(key)
        for name in TEXT_FIELDS:
            value = field(entry, name)
            if re.search(r"(?<!\\)&", value):
                failures.append(f"{key}: raw & in {name} (write \\&)")
            if "\u2014" in value:
                failures.append(f"{key}: raw em dash in {name} (write ---)")
        absent = [f for f in gaps(entry) if f not in KNOWN_INCOMPLETE.get(key, ())]
        if absent:
            failures.append(f"{key}: @{entry.type} lacks {', '.join(absent)}")
    for (kind, value), keys in owners.items():
        keys = sorted(set(keys))
        if len(keys) > 1:
            failures.append(f"{kind} {value} shared by {', '.join(keys)}")
    return failures


def defined_keys(files):
    entries, _ = load(files)
    return {key for _, key, _ in entries}


def cited(text):
    for m in CITE.finditer(text):
        for key in m.group(1).split(","):
            if key.strip():
                yield key.strip()


def sources():
    """(label, text) for tracked markdown files and notebook markdown cells."""
    for path in tracked("*.md"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("tests/", ".claude/")):
            continue
        yield rel, path.read_text(encoding="utf-8")
    for path in tracked("*.ipynb"):
        rel = path.relative_to(ROOT).as_posix()
        nb = json.loads(path.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                src = cell.get("source", "")
                yield f"{rel}:cell{i}", "".join(src) if isinstance(src, list) else src


def test_bibliography_form():
    failures = check_bibliography(bib_files())
    assert not failures, "\n".join(failures)


def undefined_citations(keys, texts):
    """'<label>: <key>' for each key cited in the (label, text) pairs `texts` that is not in `keys`."""
    return [f"{label}: {key}" for label, text in texts for key in cited(text) if key not in keys]


def test_every_cited_key_exists():
    failures = undefined_citations(defined_keys(bib_files()), sources())
    assert not failures, "undefined citation keys:\n" + "\n".join(failures)


def test_a_cited_key_that_does_not_exist_is_reported():
    texts = [("a.md", "As {cite}`one` and {cite:t}`two,missing` show."), ("b.ipynb:cell1", "{cite:p}`one`")]
    assert undefined_citations({"one", "two"}, texts) == ["a.md: missing"]


def test_known_incomplete_entries_are_still_incomplete():
    """An allow-list entry that has been completed must be removed."""
    entries, _ = load(bib_files())
    by_key = {key: entry for _, key, entry in entries}
    stale = [k for k, fields in KNOWN_INCOMPLETE.items()
             if k not in by_key or not (set(fields) & set(gaps(by_key[k])))]
    assert not stale, f"remove from KNOWN_INCOMPLETE: {stale}"


BASE = """@article{one,
  title={A paper},
  author={Doe, Jane},
  journal={Some Journal},
  volume={1},
  pages={1--2},
  year={2020},
  doi={10.1000/xyz},
}
"""


@pytest.mark.parametrize("second, expect", [
    ("@article{two,\n title={Other}, author={Roe, Rick}, journal={J}, volume={2}, pages={3--4}, year={2021},"
     " doi={10.1000/XYZ},\n}\n", "doi 10.1000/xyz shared by one, two"),
    ("@article{one,\n title={Other}, author={Roe, Rick}, journal={J}, volume={2}, pages={3--4}, year={2021},"
     " doi={10.1000/abc},\n}\n", "defined in"),
    ("@misc{three,\n title={Fish & chips}, year={2021}, url={https://example.org/a},\n}\n", "raw & in title"),
    ("@misc{four,\n title={No url}, year={2021},\n}\n", "lacks url"),
    ("@misc{six,\n title={Fish and chips}, note={Part one \u2014 the fish}, year={2021}, url={https://example.org/b},\n}\n",
     "raw em dash in note"),
    ("@misc{seven,\n title={T}, publisher={Fish & Co}, year={2021}, url={https://example.org/c},\n}\n",
     "raw & in publisher"),
    ("@article{eight,\n title={Other}, author={Roe, Rick}, journal={J}, volume={2}, pages={3--4}, year={2021},"
     " doi={https://doi.org/10.1000/XYZ},\n}\n", "doi 10.1000/xyz shared by eight, one"),
    ("@misc{nine,\n title={T}, year={2021}, url={https://arxiv.org/pdf/2401.00001v2.pdf},\n}\n"
     "@misc{ten,\n title={U}, year={2021}, url={https://arxiv.org/abs/2401.00001},\n}\n",
     "eprint 2401.00001 shared by nine, ten"),
    ("@book{five,\n title={No link}, author={Roe, Rick}, publisher={P}, year={2021},\n}\n", "lacks identifier"),
])
def test_checks_catch_problems(tmp_path, second, expect):
    """The checks fail on a scratch copy with a duplicate DOI, a duplicate key, a raw & or a missing field."""
    first = tmp_path / "first.bib"
    other = tmp_path / "other.bib"
    first.write_text(BASE, encoding="utf-8")
    other.write_text(second, encoding="utf-8")
    failures = check_bibliography([first, other])
    assert any(expect in f for f in failures), failures


def test_fold_appends_a_chapter_file_and_refuses_a_duplicate(tmp_path):
    from tools.fold_bib import fold

    entry = "@misc{{{key},\n  title={{{title}}},\n  url={{https://example.org/{key}}},\n  year={{2020}}\n}}\n"
    (tmp_path / "references.bib").write_text(entry.format(key="a2020", title="First"), encoding="utf-8")
    (tmp_path / "bib").mkdir()
    (tmp_path / "bib" / "ch07.bib").write_text("% header\n\n" + entry.format(key="b2020", title="Second"))
    (tmp_path / "bib" / "ch08.bib").write_text(entry.format(key="c2020", title="{F}irst"))
    (tmp_path / "_config.yml").write_text(
        "bibtex_bibfiles:\n  - references.bib\n  - bib/ch07.bib\n  - bib/ch08.bib\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="same title as a2020"):
        fold(["bib/ch08.bib"], root=tmp_path)
    assert (tmp_path / "bib" / "ch08.bib").exists()
    fold(["bib/ch07.bib"], root=tmp_path)
    text = (tmp_path / "references.bib").read_text(encoding="utf-8")
    assert "@misc{b2020," in text and "% header" not in text
    assert "bib/ch07.bib" not in (tmp_path / "_config.yml").read_text(encoding="utf-8")
    assert not (tmp_path / "bib" / "ch07.bib").exists()
