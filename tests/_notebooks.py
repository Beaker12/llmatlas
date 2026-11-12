"""Reading, executing and comparing notebooks for the notebook tests.

execute(path, offline) runs a temporary copy of the notebook (with its data/ directory beside it) in a
kernel started from the interpreter that runs the tests, and returns the executed notebook.
compare_cells(committed, executed, shape, masks) lists the cells whose text outputs differ.
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from tests._compare import compare_output

CELL_TIMEOUT = 600


def notebooks(chapter_dir):
    return sorted((Path(chapter_dir) / "examples").glob("*.ipynb"))


def read(path):
    return nbformat.read(str(path), as_version=4)


def cell_text(cell):
    """The text a code cell shows: stream text and text/plain of each output, in order.

    Images (including the text/plain repr that accompanies one), HTML and every other mime type are ignored. An error output is shown as 'ERROR: name: value'.
    """
    parts = []
    for out in cell.get("outputs", []):
        kind = out["output_type"]
        if kind == "stream":
            parts.append(out["text"])
        elif kind in ("display_data", "execute_result"):
            data = out.get("data", {})
            text = data.get("text/plain")
            # the text/plain of an image output is its repr, e.g. "<Figure size ...>": part of the image
            if text is not None and not any(mime.startswith("image/") for mime in data):
                parts.append(text if text.endswith("\n") else text + "\n")
        elif kind == "error":
            parts.append(f"ERROR: {out['ename']}: {out['evalue']}\n")
    return "".join(parts)


def mask(text, masks):
    for pattern, placeholder in masks:
        text = pattern.sub(placeholder, text)
    return text


def compare_cells(committed, executed, masks=(), shape_only=False):
    """Failure strings, one per code cell whose text differs from the committed one.

    A cell whose metadata has "tol": <float> uses that absolute tolerance; one with
    "compare": "shape" is compared by shape only. Cells are matched by position.
    """
    failures = []
    old = [c for c in committed.cells if c.cell_type == "code"]
    new = [c for c in executed.cells if c.cell_type == "code"]
    if len(old) != len(new):
        return [f"expected {len(old)} code cells, got {len(new)}"]
    index = {id(c): i for i, c in enumerate(committed.cells)}
    for a, b in zip(old, new):
        want, got = mask(cell_text(a), masks), mask(cell_text(b), masks)
        if want.strip() and not got.strip():
            failures.append(f"cell {index[id(a)]}: no fresh output (committed output starts {want.splitlines()[0]!r})")
            continue
        if got.strip() and not want.strip():
            failures.append(f"cell {index[id(a)]}: no committed output (fresh output starts {got.splitlines()[0]!r})")
            continue
        shape = shape_only or a.metadata.get("compare") == "shape"
        reason = compare_output(got, want, shape=shape, tol=None if shape else a.metadata.get("tol"))
        if reason:
            failures.append(f"cell {index[id(a)]}: {reason}")
    return failures


def kernel_failures(committed, prefix="3.12"):
    version = committed.metadata.get("language_info", {}).get("version", "")
    if version.startswith(prefix):
        return []
    return [f"kernel metadata names Python {version or 'unknown'}, expected {prefix}.x"]


@contextlib.contextmanager
def kernel_environment(env):
    """Make `python3` resolve to this interpreter and apply `env` (None removes a variable), then undo both."""
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp, "kernels", "python3")
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python 3", "language": "python"}))
        saved = {k: os.environ.get(k) for k in [*env, "JUPYTER_PATH"]}
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.environ["JUPYTER_PATH"] = tmp
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def execute(path, offline=True, timeout=CELL_TIMEOUT):
    """Execute a temporary copy of the notebook and return the executed notebook.

    The copy sits in a fresh directory with the notebook's data/ directory beside it; that directory
    is the kernel's working directory. Errors in cells are kept as outputs. A timeout, a dead kernel or any
    other exception from the client is not raised: it is recorded (see aborted()) and the cells run so far are kept.
    """
    path = Path(path)
    env = {"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "LLMATLAS_OFFLINE": "1" if offline else None}
    with tempfile.TemporaryDirectory() as tmp, kernel_environment(env):
        shutil.copy(path, tmp)
        if (path.parent / "data").is_dir():
            shutil.copytree(path.parent / "data", Path(tmp) / "data")
        nb = read(Path(tmp) / path.name)
        current = {"cell": "?"}

        def started(cell, cell_index):
            current["cell"] = cell_index

        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                                resources={"metadata": {"path": tmp}}, on_cell_start=started)
        try:
            client.execute()
        except Exception as e:  # a timeout, a dead kernel or any other failure of the client
            first = (str(e).strip().splitlines() or [""])[0]
            nb.metadata["aborted"] = f"cell {current['cell']}: {type(e).__name__}: {first}"
        return nb


def aborted(executed):
    """'cell N: ErrorClass: message' if execution stopped early, else None."""
    return executed.metadata.get("aborted")
