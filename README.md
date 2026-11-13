# LLM Atlas

A Jupyter Book about large language models: tokenization, attention, transformers, and the mathematics of training, with runnable PyTorch examples and companion notebooks.

**Status:** chapters 1 (LLM Foundations), 2 (Mathematical Foundations), 3 (Data Acquisition and Sources), 4 (Data Preprocessing and Cleaning), 5 (Dataset Management and Versioning) and 6 (Modern Transformer Architectures) are written. The remaining chapters are planned; see `roadmap.md`.

Published at <https://beaker12.github.io/llmatlas/>.

## Local setup

Requires Python 3.12, the version the workflows use and the committed notebooks' kernel metadata names (a test checks it).

```bash
git clone https://github.com/Beaker12/llmatlas.git
cd llmatlas
make install
```

`make install` creates `.venv` with `python3.12` (override with `make install PYTHON=...`), installs the CPU build of PyTorch, then installs `requirements.txt` and `requirements-dev.txt`, all pinned by `constraints.txt`, and installs the pre-commit hooks. The manual equivalent is:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt
pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

### Pinned versions

`constraints.txt` pins every package, PyTorch included. To regenerate it after changing `requirements.txt` or `requirements-dev.txt`, run `make constraints`. It needs [uv](https://docs.astral.sh/uv/) on the `PATH` (or `make constraints UV=/path/to/uv`): it runs `uv pip compile` for Python 3.12 against PyPI and the CPU PyTorch index, keeps the header comments of the existing file, drops the `+cpu` label from the torch pin so other platforms can install it from PyPI, and replaces the file only if `uv` succeeds. uv is used for nothing else; installation uses pip.

## Run the notebooks on Binder

The demo notebooks in `chapters/*/examples/` also run on [mybinder.org](https://mybinder.org) with no local install. Binder builds its environment from `binder/requirements.txt`, which adds the CPU build of PyTorch to `requirements.txt`. Open a notebook with:

```text
https://mybinder.org/v2/gh/Beaker12/llmatlas/<ref>?labpath=<notebook path>
```

where `<ref>` is a branch, tag or commit and `<notebook path>` is relative to the repository root. For example, chapter 1's demo on `main`:

<https://mybinder.org/v2/gh/Beaker12/llmatlas/main?labpath=chapters/01-foundations/examples/attention_mechanisms_demo.ipynb>

The first launch of a new ref builds the image, which takes several minutes.

## Build and test

```bash
make test       # YAML check, pytest tests/ (everything except live tests), strict book build
make build      # build the HTML book into _build/html (warnings are errors, nitpicky, as in CI)
make serve      # build and serve on http://localhost:8000
make notebooks  # execute every notebook offline into _build/notebooks (the committed files are not changed)
make linkcheck  # check external links
make lint       # flake8 on tools/ and tests/
```

### Test suites

Two pytest markers are defined in `pytest.ini`:

- `slow`: notebook execution and figure regeneration. Every chapter notebook is executed offline in a temporary copy and its outputs compared with the committed ones, and each `make_figures.py` is run and its SVGs compared byte for byte with the committed ones.
- `live`: needs the network or is too long to run on every commit. `pytest.ini` deselects these by default (`addopts = -m "not live"`).

```bash
.venv/bin/python -m pytest tests/ -q -m "not slow and not live"   # fast suite
.venv/bin/python -m pytest tests/                                  # full suite: everything but live (what make test and CI run)
.venv/bin/python -m pytest tests/ -m live                          # live tests (notebooks executed online)
```

The `Build and deploy book` workflow runs the full suite with `LLMATLAS_OFFLINE=1`, then the strict build, and builds the PDF in a separate job. The weekly `Live checks` workflow runs the live tests, `make linkcheck` and `tools/verify_arxiv.py`.

### Environment variables

- `LLMATLAS_OFFLINE=1`: notebooks and figure scripts read the copies committed in the chapter's `examples/data/` or `figures/data/` directory instead of the network. The notebook tests, `make notebooks`, the figure scripts and the build workflow set it.
- `LLMATLAS_LIVE=1`: set by the weekly `Live checks` workflow to mark a live run. Live tests are selected with `-m live`, not by this variable.
- `LLMATLAS_USER_AGENT`: the User-Agent header for network requests made by the chapter 3 and 6 notebooks, `tools/fetch_arch_samples.py` and `tools/verify_arxiv.py`, each of which has its own default.
- `LLMATLAS_WRITE_DATA=1`: makes the network example in chapter 1's tokenization deep-dive, which prints token counts per language for several encodings, also write them to `tokens_per_language.json` in the current directory. Run it from `chapters/01-foundations/figures/data/`, where `make_figures.py` reads that file for its plot. Without the variable the example only prints.

### Re-executing a notebook for commit

The committed notebooks carry their outputs, and the published pages show them. Execute a notebook in place from its own directory, offline, with a Python 3.12 kernel and with `MPLBACKEND` unset: a non-interactive backend such as `Agg` drops the embedded plots, and tests fail on a notebook committed without them.

```bash
cd chapters/03-data-acquisition/examples
env -u MPLBACKEND LLMATLAS_OFFLINE=1 ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=600 data_acquisition_demo.ipynb
```

`tests/test_notebook_outputs.py` (slow) then checks that a fresh offline execution reproduces the committed outputs.

### Regenerating figures

Each figure is committed as `.svg` (used by the HTML book) and `.pdf` (used by the PDF build) in the chapter's `figures/` directory.

- Data plots: `python chapters/<chapter>/figures/make_figures.py <output-directory>` writes them from the committed files in `figures/data/` (and, for some chapters, from the chapter notebook's cells tagged `figure:<slug>`). The output is deterministic; copy the files into `figures/` to update them. The slow figure test runs each script and requires byte-identical SVGs.
- Schematics: each is drawn from a Graphviz `.dot` source beside it. The committed SVG is authoritative: it is not re-rendered in CI, and its header records the Graphviz version that drew it (`dot -Tsvg`). After editing a `.dot` file, re-render the SVG and its PDF counterpart and record the new hash in `figures/SOURCES.sha256` (`sha256sum *.dot > SOURCES.sha256` in that directory); a test fails when a `.dot` file's hash differs from its line there.

## Contributing

Open an issue or a pull request against `main`. Please run `make test` first. The pre-commit hooks (`pre-commit install`) check whitespace, YAML and Python formatting; they deliberately do not reformat Markdown, because generic Markdown formatters corrupt MyST math.

## Layout

```
_config.yml            Jupyter Book configuration
_toc.yml               Table of contents
intro.md               Landing page
roadmap.md             Planned chapters
bibliography.md        All references (references.bib)
chapters/              Written chapters and their example notebooks
tests/                 Repository tests (math escaping, TOC, equations, notebooks)
tools/unescape_math.py Repairs Markdown-escaped LaTeX inside math regions
.github/workflows/     Build, test and deploy to GitHub Pages
```

## Deployment

Pushes to `main` build and deploy to GitHub Pages. Pull requests are built and tested but not deployed.

## License

LLM Atlas is dual-licensed:

- **Text, figures and other content** are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). See [LICENSE-CONTENT](LICENSE-CONTENT).
- **Code**, meaning the code in the chapters and notebooks, `tools/` and `tests/`, is licensed under the MIT License. See [LICENSE](LICENSE).
