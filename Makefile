# Energy Descent Certificates — task runner.
# Every target is CPU-only and offline unless it explicitly says GPU/Modal.

PY := PYTHONPATH=src python
UV := uv

.DEFAULT_GOAL := help

.PHONY: help sync test smoke lint fmt sweep figures tables clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Create/refresh the pinned env (python 3.12)
	$(UV) sync --python 3.12 --extra dev

test: ## Offline CPU test suite (no GPU, no downloads)
	$(UV) run --extra dev env PYTHONPATH=src pytest -q

lint: ## Ruff lint
	$(UV) run --extra dev ruff check src tests experiments analysis

fmt: ## Ruff format + import sort
	$(UV) run --extra dev ruff format src tests experiments analysis
	$(UV) run --extra dev ruff check --fix src tests experiments analysis

smoke: ## End-to-end base-reasoner smoke on CPU (<60s), appends a ledger row
	$(UV) run env PYTHONPATH=src python -m edc.cli smoke --config configs/smoke.toml

sweep: ## Expand + run a sweep grid: make sweep CONFIG=configs/sweeps/<file>.toml
	$(UV) run env PYTHONPATH=src python experiments/run_sweep.py $(CONFIG)

figures: ## Regenerate all paper figures from results/ledger.jsonl
	$(UV) run --extra plot env PYTHONPATH=src python analysis/make_figures.py

tables: ## Regenerate all paper tables from results/ledger.jsonl
	$(UV) run env PYTHONPATH=src python analysis/make_tables.py

clean: ## Remove caches and generated (regenerable) artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/raw data/processed
