"""Build tests/spelling_uk_us.txt and tests/spelling_house.txt (not a test; run by hand).

    python tests/make_spelling_list.py varcon.txt README tests/spelling_uk_us.txt tests/spelling_house.txt

varcon.txt and README come from https://github.com/en-wl/wordlist (directory varcon/); README holds the version
(2020.12.07) and the licence notice that is copied into the header of the first output.

spelling_uk_us.txt. Each data line of varcon.txt is `TAGS: word / TAGS: word ...` (an optional ` | usage` suffix is
dropped). A tag is a category (A American, B British -ise, Z British -ize, C Canadian, D Australian) with an optional
variant mark (`.` equal, `v` variant, `V` seldom used variant, `-` possible, `x` improper).
- American spellings: every word carrying an A tag with a mark other than `x` or `-` (so `Av` counts), on any line.
- A pair is (word with a B or Z tag, the line's preferred American word), kept when the British word is a single
  lowercase word, is not an American spelling by the rule above, and the two differ.
- A British word with two different American partners is dropped (ambiguous).
Words VarCon accepts in American English (grey, catalogue, judgement ...) are therefore not in this list.

spelling_house.txt, this book's supplement, in the same format:
(a) for every VarCon verb pair -ise/-ize and -yse/-yze, the inflections VarCon lacks: -ised, -ises, -ising, -isation,
    -isations, -iser, -isers (for -yse verbs without the -yses form, which is also the plural of "analysis");
(b) the same inflections for the domain verbs in DOMAIN_VERBS that VarCon lacks;
(c) HOUSE_CHOICES, spellings VarCon accepts in American English but the book writes one way.
Only forms absent from VarCon's pairs and not American spellings in VarCon are added.
"""
import re
import sys
from collections import defaultdict

TAG = re.compile(r"^([A-DZ_])([.vVx-]?)$")
SUFFIXES = [("ised", "ized"), ("ises", "izes"), ("ising", "izing"), ("isation", "ization"), ("isations", "izations"),
            ("iser", "izer"), ("isers", "izers")]
DOMAIN_VERBS = ["tokenise", "detokenise", "pretokenise", "subtokenise", "pseudonymise", "binarise", "lemmatise",
                "quantise", "vectorise", "discretise", "canonicalise", "regularise", "parametrise", "reparametrise",
                "linearise", "diagonalise", "orthogonalise", "marginalise", "factorise", "memoise", "sanitise"]
HOUSE_CHOICES = {"catalogue": "catalog", "catalogues": "catalogs", "encyclopaedia": "encyclopedia",
                 "encyclopaedias": "encyclopedias", "encyclopaedic": "encyclopedic", "grey": "gray", "greys": "grays",
                 "judgement": "judgment", "judgements": "judgments", "acknowledgement": "acknowledgment",
                 "acknowledgements": "acknowledgments", "ageing": "aging", "learnt": "learned", "spelt": "spelled",
                 "whilst": "while", "amongst": "among"}


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
    """(pairs, american): the British-to-American pairs and the set of words VarCon accepts in American English."""
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
    return dict(sorted(pairs.items())), american


def inflections(british_stem, american_stem, yse=False):
    """[(british, american)] forms of a verb given its stem before -ise/-ize (or -yse/-yze with yse=True)."""
    forms = []
    for uk, us in SUFFIXES:
        if yse:
            if uk == "ises":
                continue
            uk, us = uk.replace("is", "ys", 1), us.replace("iz", "yz", 1)
        forms.append((british_stem + uk, american_stem + us))
    return forms


def build_house(pairs, american):
    """(derived, domain, choices): three dicts of pairs absent from `pairs` and not American spellings."""
    def fresh(candidates, taken):
        return {uk: us for uk, us in candidates if uk not in pairs and uk not in american and uk not in taken
                and re.fullmatch(r"[a-z]+", uk) and re.fullmatch(r"[a-z]+", us)}

    derived = {}
    for uk, us in pairs.items():
        for ending_uk, ending_us, yse in (("ise", "ize", False), ("yse", "yze", True)):
            if len(uk) > 5 and uk.endswith(ending_uk) and us.endswith(ending_us) and uk[:-3] == us[:-3]:
                derived.update(fresh(inflections(uk[:-3], us[:-3], yse), derived))
    domain = {}
    for verb in DOMAIN_VERBS:
        stem = verb[:-3]
        domain.update(fresh([(verb, stem + "ize")] + inflections(stem, stem), {**derived, **domain}))
    choices = {uk: us for uk, us in HOUSE_CHOICES.items() if uk not in pairs}
    return derived, domain, choices


def licence(readme):
    text = open(readme, encoding="utf-8").read()
    return text[text.index("Copyright\n=========") + len("Copyright\n========="):].strip("\n")


def main(varcon, readme, out, house_out):
    pairs, american = build(varcon)
    header = ["British to American spelling pairs, one `british<TAB>american` pair per line.",
              "Source: VarCon (variant conversion info distributed with SCOWL), version 2020.12.07,",
              "varcon/varcon.txt in https://github.com/en-wl/wordlist. Derived by tests/make_spelling_list.py:",
              "pairs whose British form is not also an accepted American spelling in VarCon. Words VarCon accepts in",
              "American English (grey, catalogue, judgement ...) are not here; tests/spelling_house.txt adds the house",
              "choices and the inflections VarCon lacks.",
              "", "VarCon licence notice, reproduced as the licence requires:", ""]
    lines = [f"# {l}".rstrip() for l in header + licence(readme).splitlines()]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.writelines(f"{uk}\t{us}\n" for uk, us in pairs.items())
    derived, domain, choices = build_house(pairs, american)
    with open(house_out, "w", encoding="utf-8") as f:
        f.write("# This book's supplement to tests/spelling_uk_us.txt, maintained by hand; same format, loaded with it.\n"
                "# Sections (a) and (b) were generated by tests/make_spelling_list.py from VarCon's -ise/-ize and\n"
                "# -yse/-yze verb pairs and the DOMAIN_VERBS list in that script; section (c) is house style where\n"
                "# VarCon accepts both spellings in American English. Edit the entries; the script is a starting point.\n")
        for title, section in (("(a) inflections of VarCon verb pairs that VarCon lacks", derived),
                               ("(b) domain verbs and their inflections", domain),
                               ("(c) house choices", choices)):
            f.write(f"\n# {title}: {len(section)} entries\n")
            f.writelines(f"{uk}\t{us}\n" for uk, us in sorted(section.items()))
    print(len(pairs), "VarCon pairs;", len(derived), "derived,", len(domain), "domain,", len(choices), "house choices")


if __name__ == "__main__":
    main(*sys.argv[1:5])
