# LLM Atlas: local development commands

.PHONY: help install build build-lenient serve clean test lint linkcheck notebooks status constraints

VENV_DIR   = .venv
PYTHON     = $(VENV_DIR)/bin/python
PIP        = $(VENV_DIR)/bin/pip
JB         = $(VENV_DIR)/bin/jupyter-book
UV         ?= uv
NOTEBOOKS  = $(wildcard chapters/*/examples/*.ipynb)

help:
	@echo "make install       create .venv and install torch (CPU) + requirements"
	@echo "make test          YAML check + pytest + strict book build + notebook execution"
	@echo "make build         build the HTML book (warnings are errors, as in CI)"
	@echo "make build-lenient build without -W"
	@echo "make serve         build and serve on http://localhost:8000"
	@echo "make notebooks     execute every notebook headlessly"
	@echo "make linkcheck     check external links"
	@echo "make lint          flake8 on Python files"
	@echo "make constraints   regenerate constraints.txt with uv (CPU torch)"
	@echo "make clean         remove build artifacts"
	@echo "make status        show branch and build state"

$(VENV_DIR)/pyvenv.cfg:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

install: $(VENV_DIR)/pyvenv.cfg
	$(PIP) install torch --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt
	$(PIP) install -r requirements.txt -r requirements-dev.txt -c constraints.txt
	$(VENV_DIR)/bin/pre-commit install
	@echo "Installed. Run 'make test'."

build:
	$(JB) build . --builder html -W --keep-going

build-lenient:
	$(JB) build . --builder html

serve: build
	@echo "Serving http://localhost:8000"
	@cd _build/html && python3 -m http.server 8000

clean:
	rm -rf _build/ .jupyter_cache/

test:
	$(PYTHON) -c "import yaml; yaml.safe_load(open('_config.yml')); yaml.safe_load(open('_toc.yml'))"
	$(PYTHON) -m pytest tests/
	$(MAKE) build
	$(MAKE) notebooks

notebooks:
	@for nb in $(NOTEBOOKS); do \
	  echo "Executing $$nb"; \
	  LLMATLAS_OFFLINE=1 $(VENV_DIR)/bin/jupyter nbconvert --to notebook --execute \
	    --ExecutePreprocessor.timeout=600 --output-dir _build/notebooks $$nb || exit 1; \
	done

linkcheck:
	$(JB) build . --builder linkcheck

lint:
	$(VENV_DIR)/bin/flake8 tools tests --max-line-length=100

status:
	@echo "Branch: $(shell git branch --show-current)"
	@git status --porcelain
	@echo "Built book: $(shell test -f _build/html/index.html && echo present || echo missing)"

# torch comes in on stdin so no extra input file is needed. The "+cpu" local
# label is dropped from the pin: PEP 440 still matches the CPU wheel on Linux,
# and macOS/Windows can satisfy the same constraint from PyPI.
# unsafe-best-match: uv otherwise takes each package only from the first index
# that has it, which would hide the CPU torch wheel when PyPI is also searched.
# The output is built in temporary files so a failed uv never clobbers the file.
constraints:
	@command -v $(UV) >/dev/null || { echo "error: uv not found (UV=$(UV))" >&2; exit 1; }
	@printf 'torch\n' | $(UV) pip compile requirements.txt requirements-dev.txt - \
	    --python-version 3.12 --extra-index-url https://download.pytorch.org/whl/cpu \
	    --index-strategy unsafe-best-match --no-header > constraints.body \
	  || { rm -f constraints.body; echo "error: uv pip compile failed" >&2; exit 1; }
	@test -s constraints.body || { rm -f constraints.body; echo "error: uv produced no output" >&2; exit 1; }
	@{ sed '/^[^#]/,$$d' constraints.txt; \
	   python3 -c 'import sys; c="# torch is pinned without the +cpu label so non-Linux platforms can use PyPI;\n# PEP 440 still matches the +cpu wheel from the CPU index on Linux.\n"; [sys.stdout.write(c+l[:-5]+"\n" if l.startswith("torch==") and l.endswith("+cpu\n") else l) for l in sys.stdin]' < constraints.body; \
	 } > constraints.tmp && mv constraints.tmp constraints.txt; rc=$$?; rm -f constraints.body constraints.tmp; exit $$rc
