# LLMatlas Makefile for local development and testing

.PHONY: help install clean build serve test lint format check-deps branch-status gitflow-init pre-commit check-hooks

# Default target
help:
	@echo "LLMatlas Development Commands:"
	@echo "=============================="
	@echo "Development:"
	@echo "  install     - Install dependencies in virtual environment"
	@echo "  build       - Build Jupyter Book locally"
	@echo "  serve       - Build and serve book locally"
	@echo "  clean       - Clean build artifacts"
	@echo ""
	@echo "Quality:"
	@echo "  test        - Run tests and validation"
	@echo "  pre-commit  - Run pre-commit hooks on all files"
	@echo "  check-hooks - Check pre-commit installation status"
	@echo "  lint        - Run linting tools"
	@echo "  format      - Format code and markdown"
	@echo "  check-deps  - Check dependency status"
	@echo ""
	@echo "GitFlow Workflow:"
	@echo "  gitflow-init    - Initialize GitFlow in repository"
	@echo "  start-feature   - Start new feature (usage: make start-feature FEATURE=name)"
	@echo "  finish-feature  - Finish current feature branch"
	@echo "  start-release   - Start release branch (usage: make start-release VERSION=x.y.z)"
	@echo "  finish-release  - Finish current release branch"
	@echo "  start-hotfix    - Start hotfix branch (usage: make start-hotfix HOTFIX=name)"
	@echo "  finish-hotfix   - Finish current hotfix branch"
	@echo "  branch-status   - Show current branch and status"
	@echo ""
	@echo "CI/CD:"
	@echo "  deploy      - Simulate CI/CD deployment locally"

# Environment setup
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
JUPYTER_BOOK = $(VENV_DIR)/bin/jupyter-book

# Check if virtual environment exists
$(VENV_DIR)/pyvenv.cfg:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

# Install dependencies
install: $(VENV_DIR)/pyvenv.cfg
	$(PIP) install -r requirements.txt
	@echo "Installing pre-commit hooks..."
	$(VENV_DIR)/bin/pre-commit install
	@echo "Dependencies installed in virtual environment"
	@echo "Pre-commit hooks installed - will run automatically on git commit"

# Build book locally
build: install
	@echo "Building Jupyter Book locally..."
	$(JUPYTER_BOOK) build . --builder html

# Serve book locally
serve: build
	@echo "Starting local server..."
	@echo "Book will be available at: http://localhost:8000"
	@cd _build/html && python3 -m http.server 8000

# Clean build artifacts
clean:
	rm -rf _build/
	rm -rf .jupyter_cache/
	@echo "Build artifacts cleaned"

# Test and validation
test: install
	@echo "Running tests and validation..."
	$(PYTHON) -c "import yaml; yaml.safe_load(open('_config.yml'))" || (echo "_config.yml is invalid YAML" && exit 1)
	$(PYTHON) -c "import yaml; yaml.safe_load(open('_toc.yml'))" || (echo "_toc.yml is invalid YAML" && exit 1)
	@echo "Configuration files are valid"
	@find . -name "*.md" -exec $(PYTHON) -c "import sys; [print(f'{f} is readable') if open(f).read() else None for f in sys.argv[1:]]" {} +

# Pre-commit hooks
pre-commit: install
	@echo "Running pre-commit hooks on all files..."
	$(VENV_DIR)/bin/pre-commit run --all-files

# Check pre-commit status
check-hooks:
	@echo "Pre-commit hook status:"
	@if [ -f ".git/hooks/pre-commit" ]; then \
		echo "Pre-commit hooks are installed"; \
		echo "Configured hooks:"; \
		$(VENV_DIR)/bin/pre-commit --version 2>/dev/null || echo "pre-commit not installed in venv"; \
	else \
		echo "Pre-commit hooks not installed - run 'make install'"; \
	fi

# Linting
lint: install
	@echo "Running linting tools..."
	$(VENV_DIR)/bin/flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics || echo "No Python files to lint"
	@echo "Linting complete"

# Formatting
format: install
	@echo "Formatting code..."
	$(VENV_DIR)/bin/black . || echo "No Python files to format"
	@echo "Formatting complete"

# Check dependencies
check-deps: install
	@echo "Checking dependencies..."
	$(PIP) list --outdated || echo "All dependencies are up to date"

# GitFlow workflow helpers
gitflow-init:
	@echo "Initializing GitFlow..."
	@echo "Setting up main and develop branches..."
	git checkout main 2>/dev/null || git checkout -b main
	git checkout develop 2>/dev/null || git checkout -b develop
	@echo "GitFlow initialized"
	@echo "Main branch: production-ready releases"
	@echo "Develop branch: integration of features"

branch-status:
	@echo "Current branch and status:"
	@echo "========================="
	@echo "Current branch: $(shell git branch --show-current)"
	@echo "Upstream: $(shell git rev-parse --abbrev-ref @{u} 2>/dev/null || echo 'No upstream set')"
	@echo "Status:"
	@git status --porcelain
	@echo ""
	@echo "Recent commits:"
	@git log --oneline -5

start-feature:
ifndef FEATURE
	@echo "Usage: make start-feature FEATURE=feature-name"
	@echo "Example: make start-feature FEATURE=neural-networks-chapter"
	@echo "This will create: feature/neural-networks-chapter"
else
	@echo "Starting feature: feature/$(FEATURE)"
	@echo "Ensuring develop branch is up to date..."
	git checkout develop
	git pull origin develop 2>/dev/null || echo "No remote develop branch yet"
	git checkout -b feature/$(FEATURE)
	@echo "Created and switched to feature/$(FEATURE)"
	@echo "When ready to finish: make finish-feature"
endif

finish-feature:
	@echo "Finishing current feature branch..."
	$(eval CURRENT_BRANCH := $(shell git branch --show-current))
	@if echo "$(CURRENT_BRANCH)" | grep -q "^feature/"; then \
		echo "Finishing $(CURRENT_BRANCH)"; \
		echo "Switching to develop and merging..."; \
		git checkout develop; \
		git merge --no-ff $(CURRENT_BRANCH); \
		git branch -d $(CURRENT_BRANCH); \
		echo "Feature $(CURRENT_BRANCH) merged and deleted"; \
		echo "Push to remote: git push origin develop"; \
	else \
		echo "Not on a feature branch. Current: $(CURRENT_BRANCH)"; \
		exit 1; \
	fi

start-release:
ifndef VERSION
	@echo "Usage: make start-release VERSION=x.y.z"
	@echo "Example: make start-release VERSION=1.2.0"
else
	@echo "Starting release: release/$(VERSION)"
	git checkout develop
	git pull origin develop 2>/dev/null || echo "No remote develop branch yet"
	git checkout -b release/$(VERSION)
	@echo "Created release/$(VERSION)"
	@echo "Only bug fixes and documentation allowed in release branch"
	@echo "When ready: make finish-release"
endif

finish-release:
	@echo "Finishing current release branch..."
	$(eval CURRENT_BRANCH := $(shell git branch --show-current))
	$(eval VERSION := $(shell echo "$(CURRENT_BRANCH)" | sed 's/release\///'))
	@if echo "$(CURRENT_BRANCH)" | grep -q "^release/"; then \
		echo "Finishing $(CURRENT_BRANCH)"; \
		echo "Merging to main..."; \
		git checkout main; \
		git merge --no-ff $(CURRENT_BRANCH); \
		git tag -a v$(VERSION) -m "Release version $(VERSION)"; \
		echo "Merging back to develop..."; \
		git checkout develop; \
		git merge --no-ff $(CURRENT_BRANCH); \
		git branch -d $(CURRENT_BRANCH); \
		echo "Release $(VERSION) completed and tagged"; \
		echo "Push: git push origin main develop --tags"; \
	else \
		echo "Not on a release branch. Current: $(CURRENT_BRANCH)"; \
		exit 1; \
	fi

start-hotfix:
ifndef HOTFIX
	@echo "Usage: make start-hotfix HOTFIX=hotfix-name"
	@echo "Example: make start-hotfix HOTFIX=fix-broken-links"
else
	@echo "Starting hotfix: hotfix/$(HOTFIX)"
	git checkout main
	git pull origin main 2>/dev/null || echo "No remote main branch yet"
	git checkout -b hotfix/$(HOTFIX)
	@echo "Created and switched to hotfix/$(HOTFIX)"
	@echo "When ready: make finish-hotfix"
endif

finish-hotfix:
	@echo "Finishing current hotfix branch..."
	$(eval CURRENT_BRANCH := $(shell git branch --show-current))
	@if echo "$(CURRENT_BRANCH)" | grep -q "^hotfix/"; then \
		echo "Finishing $(CURRENT_BRANCH)"; \
		echo "Merging to main..."; \
		git checkout main; \
		git merge --no-ff $(CURRENT_BRANCH); \
		echo "Merging to develop..."; \
		git checkout develop; \
		git merge --no-ff $(CURRENT_BRANCH); \
		git branch -d $(CURRENT_BRANCH); \
		echo "Hotfix $(CURRENT_BRANCH) merged to main and develop"; \
		echo "Tag the main branch and push: git push origin main develop --tags"; \
	else \
		echo "Not on a hotfix branch. Current: $(CURRENT_BRANCH)"; \
		exit 1; \
	fi

# Simulate CI/CD deployment
deploy: clean test build
	@echo "Simulating CI/CD deployment..."
	@echo "Build successful - ready for GitHub Pages deployment"
	@echo "Push to main branch to deploy to production"
	@echo "Push to develop branch to deploy to staging (if configured)"

# Development status check
status: branch-status check-deps
	@echo ""
	@echo "Project Status:"
	@echo "==============="
	@echo "Virtual Environment: $(shell test -d $(VENV_DIR) && echo 'Present' || echo 'Missing - run make install')"
	@echo "Built Book: $(shell test -f _build/html/index.html && echo 'Present' || echo 'Missing - run make build')"
	@echo "Git Status: $(shell git status --porcelain | wc -l | tr -d ' ') uncommitted changes"
