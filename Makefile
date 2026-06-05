# ----------------------------
# Global project metadata

PYMODULE := src
TESTS := tests
PACKAGE_NAME := ploonetide

POETRY := $(shell command -v poetry 2>/dev/null)
CONDA := $(shell command -v conda 2>/dev/null)

CMD := $(if $(POETRY),poetry run,python -m)
INSTALL_CMD := $(if $(POETRY),poetry install --with dev --extras "dev",pip install -e .[dev])
VERSION := $(shell grep '^current_version =' pyproject.toml | sed -E 's/.*"([^"]+)"/\1/')
TAG := $(VERSION)

.PHONY: all install shell env test pytest coverage flake8 black mypy isort \
        lint format check-format clean setup.py \
        bump bump-minor bump-major release \
        export-conda-env docs  # ← added docs target here

# ----------------------------
# Installation & Environment Setup

install:
	@echo "🔍 Checking for Poetry..."
	@if ! command -v poetry >/dev/null 2>&1; then \
		echo "⚠️ Poetry not found. Installing..."; \
		pip install --user poetry; \
	else \
		echo "✅ Poetry found."; \
	fi

	@echo "🔍 Checking for active environment..."
	@if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_PREFIX" ]; then \
		echo "⚠️ No active environment detected."; \
		echo "💡 Create one now?"; \
		echo "   [1] Poetry (recommended)"; \
		if command -v conda >/dev/null 2>&1; then echo "   [2] Conda"; fi; \
		read -p "👉 Choose [1/2/skip]: " choice; \
		if [ "$$choice" = "1" ]; then \
			poetry env use python3 || poetry env use python; \
			poetry install --with dev --extras "dev"; \
			echo "✅ Poetry environment created."; \
			echo "👉 Run: poetry shell"; \
			exit 0; \
		elif [ "$$choice" = "2" ] && command -v conda >/dev/null 2>&1; then \
			poetry export --with dev --extras "dev" --format=requirements.txt --without-hashes -o conda_requirements.txt || echo '[]' > conda_requirements.txt; \
			echo "name: ploonetide-env\nchannels:\n  - conda-forge\n  - defaults\ndependencies:\n  - python=3.11\n  - pip\n  - pip:\n    - -r conda_requirements.txt" > conda_environment.yml; \
			conda env create -f conda_environment.yml; \
			echo "✅ Conda environment 'ploonetide-env' created."; \
			echo "👉 Installing ploonetide and dev dependencies..."; \
			conda run -n ploonetide-env pip install -e .[dev]; \
			rm -f conda_requirements.txt conda_environment.yml; \
			echo "✅ Environment ready. Run: conda activate ploonetide-env"; \
			exit 0; \
		else \
			echo "⏭️ Skipping. Make sure to activate an environment manually."; \
		fi \
	else \
		echo "✅ Environment active: $$VIRTUAL_ENV$$CONDA_PREFIX"; \
		$(INSTALL_CMD); \
	fi

env:
	@echo "🔍 Python:       $$(which python)"
	@echo "🔍 Poetry:       $(POETRY)"
	@echo "🔍 Conda:        $(CONDA)"
	@echo "🔍 VIRTUAL_ENV:  $$VIRTUAL_ENV"
	@echo "🔍 CONDA_PREFIX: $$CONDA_PREFIX"

export-conda-env:
	@echo "📦 Exporting conda_environment.yml from pyproject.toml..."
	poetry export --with dev --extras "dev" --format=requirements.txt --without-hashes -o conda_requirements.txt
	@echo "name: ploonetide-env\nchannels:\n  - conda-forge\n  - defaults\ndependencies:\n  - python=3.11\n  - pip\n  - pip:\n    - -r conda_requirements.txt" > conda_environment.yml
	@echo "✅ conda_environment.yml created."

# ----------------------------
# Code quality & testing

all: mypy pytest flake8

pytest: ; $(CMD) pytest $(PYMODULE) $(TESTS)
coverage: ; $(CMD) pytest --cov=$(PYMODULE) $(TESTS) --cov-report html
flake8: ; $(CMD) flake8 $(PYMODULE) $(TESTS)
mypy: ; $(CMD) mypy $(PYMODULE) $(TESTS)
black: ; $(CMD) black $(PYMODULE) $(TESTS)
isort: ; $(CMD) isort $(PYMODULE) $(TESTS)
lint: flake8 mypy
format: black isort
check-format: ; $(CMD) black --check $(PYMODULE) $(TESTS)

# ----------------------------
# Versioning & releasing

## Bump patch version and tag Git as X.Y.Z
bump:		; $(CMD) bumpver update --patch

## Bump minor version and tag Git as X.Y.Z
bump-minor:	; $(CMD) bumpver update --minor

## Bump major version and tag Git as X.Y.Z
bump-major:	; $(CMD) bumpver update --major

## Push the new tag and commit to GitHub, after confirmation
release:
	@echo "✅ Ready to push version: $(VERSION) → Tag: $(TAG)"
	@if ! git rev-parse "$(TAG)" >/dev/null 2>&1; then \
		echo "❌ Tag '$(TAG)' not found. Did you run make bump first?"; \
		exit 1; \
	fi
	@read -p "Push release to GitHub (y/N)? " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		git push origin && git push origin $(TAG); \
	else \
		echo "❌ Release aborted."; \
	fi

# ----------------------------
# Maintenance

# Path configuration
CMD := poetry run
DOCS_DIR := docs
SRC_DIR := src/ploonetide
SOURCE_DIR := $(DOCS_DIR)/source
PAGES_DIR := $(SOURCE_DIR)/pages
API_DIR := $(PAGES_DIR)/api
AUTOSUMMARY_DIR := $(SOURCE_DIR)/_autosummary

clean:
	rm -rf \
		dist/ build/ *.egg-info .pytest_cache htmlcov \
		conda_requirements.txt conda_environment.yml \
		poetry.lock \
		$(DOCS_DIR)/_build \
		$(AUTOSUMMARY_DIR) \
		$(AUTOSUMMARY_PAGES) \
		$(API_DIR) \
		**/__pycache__ \
		**/*.py[cod] \
		**/.ipynb_checkpoints \
		*.log *.tmp

# ----------------------------
# Documentation

docs:
	@echo "🧹 Cleaning old autosummary and API files..."
	rm -rf $(AUTOSUMMARY_DIR) $(API_DIR)

	@echo "🌐 Starting live documentation server at http://127.0.0.1:8000"
	@sh -c "sleep 5 && python -c 'import webbrowser; webbrowser.open(\"http://127.0.0.1:8000\")'" &

	@echo "📦 Running sphinx-apidoc to generate API .rst files..."
	@$(CMD) sphinx-apidoc -o $(SOURCE_DIR)/pages/api $(SRC_DIR) --force --separate

	@echo "📄 Generating autosummary stubs for index.rst..."
	@$(CMD) sphinx-autogen -o $(SOURCE_DIR)/_autosummary $(SOURCE_DIR)/index.rst

	@echo "🔧 Building docs with sphinx-autobuild..."
	@$(CMD) sphinx-autobuild $(SOURCE_DIR) docs/_build/html
