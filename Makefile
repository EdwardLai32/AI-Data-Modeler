# AutoML Architect — development tasks.
#
# `make help` lists everything. Targets assume an activated virtualenv, or set
# PY to point at one:
#
#   make test PY=.venv/Scripts/python.exe      # Windows
#   make test PY=.venv/bin/python              # POSIX

PY      ?= python
PIP     := $(PY) -m pip
PYTEST  := $(PY) -m pytest
RUFF    := $(PY) -m ruff
MYPY    := $(PY) -m mypy
AMLA    := $(PY) -m automl_architect.cli

PKG      := automl_architect
EXAMPLES := examples
DATASET  ?= examples/churn.csv
TARGET   ?= churned

.DEFAULT_GOAL := help
.PHONY: help install install-all install-dev examples check test test-fast test-unit \
        test-integration test-live coverage lint format typecheck doctor run profile \
        serve ui docker docker-up docker-down docker-logs clean clean-workspace \
        distclean

# -----------------------------------------------------------------------------
# help
# -----------------------------------------------------------------------------

help: ## Show this help
	@echo "AutoML Architect — make targets"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: PY=$(PY)  DATASET=$(DATASET)  TARGET=$(TARGET)"

# -----------------------------------------------------------------------------
# install
# -----------------------------------------------------------------------------

install: ## Install the package with base dependencies
	$(PIP) install -e .

install-dev: ## Install with dev tooling (pytest, ruff, mypy)
	$(PIP) install -e ".[dev]"

install-all: ## Install every optional extra plus dev tooling
	$(PIP) install -e ".[dev,boost,tuning,explain,charts,reports,api,sql]"

# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------

examples: ## Regenerate the example CSVs (fixed seed, reproducible)
	$(PY) $(EXAMPLES)/generate_datasets.py

# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------

check: lint typecheck test ## Lint, typecheck, and run the full suite

test: ## Run the whole suite (fully offline, no credentials needed)
	$(PYTEST)

test-fast: ## Skip the end-to-end runs
	$(PYTEST) -m "not slow" -q

test-unit: ## Contracts, profiling, ingestion, execution — no full pipeline
	$(PYTEST) tests/test_schemas.py tests/test_profiling.py \
	          tests/test_profiling_engine.py \
	          tests/test_ingestion.py tests/test_execution.py -q

test-integration: ## Agents, orchestrator, API, reporting — full pipeline, fake LLM
	$(PYTEST) tests/test_agents.py tests/test_orchestrator.py \
	          tests/test_api.py tests/test_reporting.py -q

test-live: ## Run the tests that call the real API. Costs money.
	AUTOML_LIVE_TESTS=1 $(PYTEST) -m live -q

coverage: ## Run with a coverage report
	$(PYTEST) --cov=$(PKG) --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

# -----------------------------------------------------------------------------
# quality
# -----------------------------------------------------------------------------

lint: ## Check style
	$(RUFF) check .

format: ## Fix what can be fixed automatically
	$(RUFF) check --fix .
	$(RUFF) format .

typecheck: ## Static types
	$(MYPY) $(PKG)

# -----------------------------------------------------------------------------
# running it
# -----------------------------------------------------------------------------

doctor: ## Diagnose the install: credentials, storage, optional features
	$(AMLA) doctor

profile: ## Profile DATASET and print the measured facts. No model calls, no cost.
	$(AMLA) profile $(DATASET) --target $(TARGET)

run: ## Full run against DATASET
	$(AMLA) run $(DATASET) --target $(TARGET) --format md,html

serve: ## Serve the API on :8000
	$(AMLA) serve

ui: ## Run the dashboard dev server on :3000
	cd frontend && npm install && npm run dev

# -----------------------------------------------------------------------------
# docker
# -----------------------------------------------------------------------------

docker: ## Build the API image
	docker build -t automl-architect .

docker-up: ## Start api + postgres + frontend
	docker compose up --build -d
	@echo "dashboard http://localhost:3000   api http://localhost:8000/docs"

docker-down: ## Stop the stack (volumes survive)
	docker compose down

docker-logs: ## Follow the API logs
	docker compose logs -f api

# -----------------------------------------------------------------------------
# cleaning
# -----------------------------------------------------------------------------

clean: ## Remove caches and build artifacts
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in str(p) and 'node_modules' not in str(p)]"
	$(PY) -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'build', 'dist')]"

clean-workspace: ## Delete every stored run, artifact, and the local database
	@echo "This deletes ./workspace — every stored run, model, and report."
	@printf "Type 'yes' to continue: " && read answer && [ "$$answer" = "yes" ]
	$(PY) -c "import shutil; shutil.rmtree('workspace', ignore_errors=True)"

distclean: clean ## clean, plus egg-info and frontend build output
	$(PY) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
	$(PY) -c "import shutil; shutil.rmtree('frontend/.next', ignore_errors=True)"
