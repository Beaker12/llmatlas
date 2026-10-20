# LLM Atlas

A Jupyter Book about large language models: tokenization, attention, transformers, and the mathematics of training, with runnable PyTorch examples and companion notebooks.

**Status:** chapters 1 (LLM Foundations), 2 (Mathematical Foundations), 3 (Data Acquisition and Sources) and 4 (Data Preprocessing and Cleaning) are written. The remaining chapters are planned; see `roadmap.md`.

Published at <https://beaker12.github.io/llmatlas/>.

## Local setup

Requires Python 3.11 or later.

```bash
git clone https://github.com/Beaker12/llmatlas.git
cd llmatlas
make install
```

`make install` creates `.venv`, installs the CPU build of PyTorch, then installs `requirements.txt` and `requirements-dev.txt`. The manual equivalent is:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
```

## Build and test

```bash
make test     # YAML check, pytest, strict book build, notebook execution
make build    # build the HTML book into _build/html
make serve    # build and serve on http://localhost:8000
make linkcheck
```

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

MIT. See [LICENSE](LICENSE).
