"""tools/list_exemptions.py on a copy of the sample chapter given one exemption of each kind."""
import json
import shutil
import subprocess
import sys

from tests._files import CHAPTER_DIRS, ROOT
from tools.list_exemptions import exemptions, report_lines

FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
KINDS = ["SYNC_EXEMPT", "NOT_A_RESULT", "NOTATION_OK", "INTERFACE_LABELS", "Also means", "lifted no-meaning ban",
         "spelling-allow.txt", "proper-nouns.txt", "static-numbers", "tol/compare", "paper note"]


def chapter_with_every_kind(tmp_path):
    chapter = tmp_path / "sample-chapter"
    shutil.copytree(FIXTURE, chapter)
    constants = tmp_path / "constants.py"
    constants.write_text('SYNC_EXEMPT: dict[str, str] = {"helper": "the notebook logs too"}\n'
                         'NOT_A_RESULT = {r"parses two files": "a count of inputs"}\n'
                         'NOTATION_OK: dict[str, str] = {r"capacity factor": "quoted"}\n'
                         'INTERFACE_LABELS: dict[str, str] = {"sec-ch0-intro": "Introduction"}\n'
                         'MIN_FIGURES: int = 2\n', encoding="utf-8")
    (chapter / "notation-local.md").write_text(
        "| Symbol | Meaning | First used | Rename | Also means |\n|---|---|---|---|---|\n"
        "| $w$ | shingle width | index.md:L1 |  | given weight of source s; never side by side |\n"
        "| $\\Xi$ | a fixture symbol | index.md:L1 |  |  |\n", encoding="utf-8")
    (chapter / "spelling-allow.txt").write_text("# names\nCentre for Fixtures\n", encoding="utf-8")
    (chapter / "proper-nouns.txt").write_text("FixtureNet-2\n", encoding="utf-8")
    notebook = chapter / "examples" / "sample_demo.ipynb"
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    markdown = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown")
    code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    nb["cells"][markdown]["metadata"]["tags"] = ["static-numbers"]
    nb["cells"][code]["metadata"]["tol"] = 0.05
    notebook.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    with open(chapter / "index.md", "a", encoding="utf-8") as page:
        page.write("\nThe RoFormer paper writes the frequencies as $\\theta_i$.\n")
    return chapter, constants, markdown, code


def test_every_kind_is_listed_once_in_order(tmp_path):
    chapter, constants, markdown, code = chapter_with_every_kind(tmp_path)
    found = exemptions(chapter, constants=constants, no_meaning=[(0, "\\Xi", "write x"), (2, "w", "elsewhere")])
    assert [kind for kind, _, _ in found] == KINDS, found
    by_kind = {kind: (entry, reason) for kind, entry, reason in found}
    assert by_kind["NOT_A_RESULT"] == ("parses two files", "a count of inputs")
    assert by_kind["INTERFACE_LABELS"] == ("sec-ch0-intro", "Introduction")
    assert by_kind["Also means"] == ("w (shingle width)", "given weight of source s; never side by side")
    assert by_kind["lifted no-meaning ban"] == ("\\Xi (a fixture symbol)", "the ban said: write x")
    assert by_kind["spelling-allow.txt"][0] == "Centre for Fixtures"
    assert by_kind["static-numbers"][0].startswith(f"sample_demo.ipynb cell {markdown}: ")
    assert by_kind["tol/compare"] == (by_kind["tol/compare"][0], "tol=0.05")
    assert by_kind["tol/compare"][0].startswith(f"sample_demo.ipynb cell {code}: ")
    assert by_kind["paper note"][0].endswith("The RoFormer paper writes the frequencies as $\\theta_i$.")
    assert by_kind["paper note"][1] == "exempt: $\\theta_i$"
    lines = report_lines(found)
    assert [line for line in lines if line.startswith("== ")] == [f"== {kind} (1) ==" for kind in KINDS]


def test_the_untouched_fixture_has_no_exemptions():
    assert exemptions(FIXTURE, constants=None, no_meaning=[]) == []


def test_a_real_chapter_lists_its_interface_labels_and_runs_as_a_script():
    found = exemptions(CHAPTER_DIRS[3])
    assert ("INTERFACE_LABELS", "sec-ch4-dedup-scope", "5.3 Near-duplicate detection with MinHash and LSH") in found
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "list_exemptions.py"), str(CHAPTER_DIRS[3])],
                          capture_output=True, text=True, check=True)
    assert "== INTERFACE_LABELS (1) ==" in proc.stdout and proc.stderr.strip() == f"# {len(found)} exemptions"


def test_the_tool_imports_nothing_from_tests():
    check = ("import sys; sys.path.insert(0, '.'); import tools.list_exemptions; "
             "print(any(m == 'tests' or m.startswith('tests.') for m in sys.modules), 'pytest' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", check], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]
