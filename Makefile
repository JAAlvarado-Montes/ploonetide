# ----------------------------
# Global project metadata

PYMODULE := src
TESTS := tests
PACKAGE_NAME := ploonetide

POETRY := $(shell command -v poetry 2>/dev/null)
CONDA := $(shell command -v conda 2>/dev/null)

CMD := $(if $(POETRY),poetry run,python -m)
INSTALL_BACKEND ?=
CONDA_ENV_NAME ?= ploonetide-env
CONDA_PYTHON ?= 3.11
VERSION := $(shell grep '^current_version =' pyproject.toml | sed -E 's/.*"([^"]+)"/\1/')
TAG := $(VERSION)

.PHONY: all install shell env test pytest coverage flake8 black mypy isort \
        lint format check-format clean setup.py \
        bump bump-minor bump-major release \
        export-conda-env docs  # ← added docs target here

# ----------------------------
# Installation & Environment Setup

install:
	@echo "🔍 Checking for active environment..."
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "✅ Virtualenv active: $$VIRTUAL_ENV"; \
	elif [ -n "$$CONDA_PREFIX" ]; then \
		echo "✅ Conda environment active: $${CONDA_DEFAULT_ENV:-unknown} ($$CONDA_PREFIX)"; \
		if [ "$$CONDA_DEFAULT_ENV" = "base" ]; then \
			echo "⚠️ Conda base is active. make install will not install into base."; \
		fi; \
	else \
		echo "⚠️ No active environment detected."; \
	fi; \
	backend="$(INSTALL_BACKEND)"; \
	if [ -z "$$backend" ]; then \
		echo "💡 Choose a developer install target:"; \
		echo "   [1] Poetry virtual environment (recommended)"; \
		echo "   [2] Conda environment: $(CONDA_ENV_NAME)"; \
		current_allowed=0; \
		if [ -n "$$VIRTUAL_ENV" ] || { [ -n "$$CONDA_PREFIX" ] && [ "$$CONDA_DEFAULT_ENV" != "base" ]; }; then \
			echo "   [3] Current active environment"; \
			current_allowed=1; \
		fi; \
		echo "   [skip] Do nothing"; \
		if [ "$$current_allowed" = "1" ]; then \
			printf "👉 Choose [1/2/3/skip]: "; \
		else \
			printf "👉 Choose [1/2/skip]: "; \
		fi; \
		read choice; \
		case "$$choice" in \
			1) backend="poetry" ;; \
			2) backend="conda" ;; \
			3) \
				if [ "$$current_allowed" = "1" ]; then \
					backend="current"; \
				else \
					echo "❌ Current environment install is not available."; \
					exit 1; \
				fi ;; \
			skip|"") backend="skip" ;; \
			*) echo "❌ Unknown install choice: $$choice"; exit 1 ;; \
		esac; \
	fi; \
	case "$$backend" in \
		poetry) \
			echo "🔍 Checking for Poetry..."; \
			if ! command -v poetry >/dev/null 2>&1; then \
				echo "⚠️ Poetry not found. Installing with python -m pip --user..."; \
				python -m pip install --user poetry; \
			else \
				echo "✅ Poetry found."; \
			fi; \
			echo "📦 Installing ploonetide with Poetry..."; \
			env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
				POETRY_VIRTUALENVS_CREATE=true poetry env use python3 \
				|| env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
					POETRY_VIRTUALENVS_CREATE=true poetry env use python; \
			env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
				POETRY_VIRTUALENVS_CREATE=true poetry install --with dev --extras "dev"; \
			echo "✅ Poetry environment ready: $$(env -u VIRTUAL_ENV -u CONDA_PREFIX -u CONDA_DEFAULT_ENV poetry env info --path)"; \
			echo "👉 Run commands with: poetry run <command>"; \
			;; \
		conda) \
			if ! command -v conda >/dev/null 2>&1; then \
				echo "❌ Conda not found. Install Conda or choose INSTALL_BACKEND=poetry."; \
				exit 1; \
			fi; \
			if conda env list | awk '{print $$1}' | grep -qx "$(CONDA_ENV_NAME)"; then \
				echo "✅ Conda environment '$(CONDA_ENV_NAME)' already exists."; \
			else \
				echo "📦 Creating Conda environment '$(CONDA_ENV_NAME)'..."; \
				conda create -y -n "$(CONDA_ENV_NAME)" python="$(CONDA_PYTHON)" pip; \
			fi; \
			echo "📦 Installing ploonetide into Conda environment '$(CONDA_ENV_NAME)'..."; \
			conda run -n "$(CONDA_ENV_NAME)" python -m pip install -e ".[dev]"; \
			echo "✅ Conda environment ready."; \
			echo "👉 Activate with: conda activate $(CONDA_ENV_NAME)"; \
			;; \
		current) \
			if [ -n "$$CONDA_PREFIX" ] && [ "$$CONDA_DEFAULT_ENV" = "base" ]; then \
				echo "❌ Refusing to install into Conda base."; \
				echo "👉 Choose INSTALL_BACKEND=poetry or INSTALL_BACKEND=conda."; \
				exit 1; \
			fi; \
			if [ -z "$$VIRTUAL_ENV" ] && [ -z "$$CONDA_PREFIX" ]; then \
				echo "❌ No active environment for INSTALL_BACKEND=current."; \
				exit 1; \
			fi; \
			echo "📦 Installing ploonetide into the current active environment..."; \
			python -m pip install -e ".[dev]"; \
			echo "✅ Current environment ready."; \
			;; \
		skip) \
			echo "⏭️ Install skipped."; \
			;; \
		*) \
			echo "❌ Unknown INSTALL_BACKEND='$$backend'."; \
			echo "👉 Use poetry, conda, current, or skip."; \
			exit 1; \
			;; \
	esac
	@echo "🔎 Final environment status:"
	@$(MAKE) env

env:
	@echo "🔍 Python:       $$(which python)"
	@echo "🔍 Poetry:       $(POETRY)"
	@echo "🔍 Conda:        $(CONDA)"
	@echo "🔍 VIRTUAL_ENV:  $$VIRTUAL_ENV"
	@echo "🔍 CONDA_PREFIX: $$CONDA_PREFIX"
	@echo "🔍 CONDA_DEFAULT_ENV: $${CONDA_DEFAULT_ENV:-}"

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
DOCS_BUILD_DIR := $(DOCS_DIR)/_build/html
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

	@echo "📦 Running sphinx-apidoc to generate API .rst files..."
	@$(CMD) sphinx-apidoc -o $(API_DIR) $(SRC_DIR) --force --separate

	@echo "📄 Generating autosummary stubs for index.rst..."
	@$(CMD) sphinx-autogen -o $(AUTOSUMMARY_DIR) $(SOURCE_DIR)/index.rst

	@echo "🌐 Starting live documentation server at http://127.0.0.1:8000"
	@echo "🔧 Building docs with sphinx-autobuild..."
	@$(CMD) sphinx-autobuild \
		--open-browser \
		--watch $(SRC_DIR) \
		--ignore "$(abspath $(API_DIR))" \
		--ignore "$(abspath $(AUTOSUMMARY_DIR))" \
		$(SOURCE_DIR) $(DOCS_BUILD_DIR)
