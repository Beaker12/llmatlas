"""Build tests/spelling_uk_us.txt from VarCon (not a test; run by hand).

    python tests/make_spelling_list.py varcon.txt README tests/spelling_uk_us.txt

varcon.txt and README come from https://github.com/en-wl/wordlist (directory varcon/); README holds the version
(2020.12.07) and the licence notice that is copied into the header of the output.

Rules. Each data line of varcon.txt is `TAGS: word / TAGS: word ...` (an optional ` | usage` suffix is dropped).
A tag is a category (A American, B British -ise, Z British -ize, C Canadian, D Australian) with an optional
variant mark (`.` equal, `v` variant, `V` seldom used variant, `-` possible, `x` improper).
- American spellings: every word carrying an A tag with a mark other than `x` or `-` (so `Av` counts), on any line.
- A pair is (word with a B or Z tag, the line's preferred American word), kept when the British word is a single
  lowercase word, is not an American spelling by the rule above, and the two differ.
- A British word with two different American partners is dropped (ambiguous).
"""
import re
import sys
from collections import defaultdict

TAG = re.compile(r"^([A-DZ_])([.vVx-]?)$")


def parse(path):
    for raw in open(path, encoding="latin-1"):   # varcon.txt is ISO 8859-1
        line = raw.split(" | ")[0].strip()
        if not line or line.startswith("#"):
            continue
        entries = []
        for part in line.split(" / "):
            tags, _, word = part.partition(": ")
            marks = {}
            for tag in tags.split():
                m = TAG.match(tag)
                if m:
                    marks[m.group(1)] = m.group(2)
            entries.append((marks, word.strip()))
        yield entries


def build(path):
    american, candidates = set(), defaultdict(set)
    for entries in parse(path):
        preferred = [w for marks, w in entries if marks.get("A") in ("", ".")] or \
                    [w for marks, w in entries if marks.get("A") == "v"]
        for marks, word in entries:
            if marks.get("A") not in (None, "x", "-"):
                american.add(word)
        for marks, word in entries:
            if any(marks.get(c) not in (None, "x", "-") for c in "BZ") and preferred:
                candidates[word].add(preferred[0])
    pairs = {}
    for word, partners in candidates.items():
        if word in american or len(partners) != 1 or not re.fullmatch(r"[a-z]+", word):
            continue
        (us,) = partners
        if us != word and re.fullmatch(r"[a-z]+", us):
            pairs[word] = us
    return dict(sorted(pairs.items()))


def licence(readme):
    text = open(readme, encoding="utf-8").read()
    return text[text.index("Copyright\n=========") + len("Copyright\n========="):].strip("\n")


def main(varcon, readme, out):
    pairs = build(varcon)
    header = ["British to American spelling pairs, one `british<TAB>american` pair per line.",
              "Source: VarCon (variant conversion info distributed with SCOWL), version 2020.12.07,",
              "varcon/varcon.txt in https://github.com/en-wl/wordlist. Derived by tests/make_spelling_list.py:",
              "pairs whose British form is not also an accepted American spelling in VarCon.",
              "", "VarCon licence notice, reproduced as the licence requires:", ""]
    lines = [f"# {l}".rstrip() for l in header + licence(readme).splitlines()]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.writelines(f"{uk}\t{us}\n" for uk, us in pairs.items())
    print(len(pairs), "pairs")


if __name__ == "__main__":
    main(*sys.argv[1:4])
