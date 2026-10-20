# LLM Atlas: local development commands

.PHONY: help install build build-strict serve clean test lint linkcheck notebooks status

VENV_DIR   = .venv
PYTHON     = $(VENV_DIR)/bin/python
PIP        = $(VENV_DIR)/bin/pip
JB         = $(VENV_DIR)/bin/jupyter-book
NOTEBOOKS  = $(wildcard chapters/*/examples/*.ipynb)

help:
	@echo "make install       create .venv and install torch (CPU) + requirements"
	@echo "make test          YAML check + pytest + strict book build + notebook execution"
	@echo "make build         build the HTML book"
	@echo "make build-strict  build with warnings as errors"
	@echo "make serve         build and serve on http://localhost:8000"
	@echo "make notebooks     execute every notebook headlessly"
	@echo "make linkcheck     check external links"
	@echo "make lint          flake8 on Python files"
	@echo "make clean         remove build artifacts"
	@echo "make status        show branch and build state"

$(VENV_DIR)/pyvenv.cfg:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

install: $(VENV_DIR)/pyvenv.cfg
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(VENV_DIR)/bin/pre-commit install
	@echo "Installed. Run 'make test'."

build:
	$(JB) build . --builder html

build-strict:
	$(JB) build . --builder html -W --keep-going

serve: build
	@echo "Serving http://localhost:8000"
	@cd _build/html && python3 -m http.server 8000

clean:
	rm -rf _build/ .jupyter_cache/

test:
	$(PYTHON) -c "import yaml; yaml.safe_load(open('_config.yml')); yaml.safe_load(open('_toc.yml'))"
	$(PYTHON) -m pytest tests/
	$(MAKE) build-strict
	$(MAKE) notebooks

notebooks:
	@for nb in $(NOTEBOOKS); do \
	  echo "Executing $$nb"; \
	  $(VENV_DIR)/bin/jupyter nbconvert --to notebook --execute \
	    --ExecutePreprocessor.timeout=120 --output-dir _build/notebooks $$nb || exit 1; \
	done

linkcheck:
	$(JB) build . --builder linkcheck

lint:
	$(VENV_DIR)/bin/flake8 tools tests --max-line-length=100

status:
	@echo "Branch: $(shell git branch --show-current)"
	@git status --porcelain
	@echo "Built book: $(shell test -f _build/html/index.html && echo present || echo missing)"
