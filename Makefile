.PHONY: help test lint typecheck benchmark benchmark-quick clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

install: ## Install dependencies
	pip install -r requirements.txt

lint: ## Run ruff linter
	ruff check standalone_*.py librarian*.py kb_client.py playbook_reader.py benchmark.py

typecheck: ## Run mypy type checker
	mypy --ignore-missing-imports standalone_*.py librarian*.py

format: ## Auto-format code with ruff
	ruff format standalone_*.py librarian*.py kb_client.py playbook_reader.py benchmark.py

# ---------------------------------------------------------------------------
# Tests (for the orchestrator itself)
# ---------------------------------------------------------------------------

test: ## Run orchestrator unit tests
	python3 -m pytest tests/ -v --tb=short

test-coverage: ## Run tests with coverage
	python3 -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

benchmark: ## Run full benchmark suite (5 tasks, ~3-5 hours)
	python3 benchmark.py --suite standard

benchmark-quick: ## Run only Level 2 task (~15 min)
	python3 benchmark.py --task 1

benchmark-l5: ## Run Level 5 bookmark manager (~1-2 hours)
	python3 benchmark.py --task 4

benchmark-list: ## List available benchmark tasks
	python3 benchmark.py --list

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean: ## Remove runtime artifacts
	rm -rf __pycache__ venv *.pyc .agents/
	rm -f knowledge_base.db playbook.json benchmark_results.json

loc: ## Count lines of code (excluding tests and prompts)
	@echo "Core orchestrator:"
	@wc -l standalone_*.py | tail -1
	@echo "Support modules:"
	@wc -l librarian*.py kb_client.py playbook_reader.py | tail -1
	@echo "Subconscious daemon:"
	@wc -l subconscious-daemon/*.py | tail -1
	@echo "Prompts:"
	@wc -l prompts/*.txt | tail -1
	@echo "---"
	@echo "Total:"
	@wc -l standalone_*.py librarian*.py kb_client.py playbook_reader.py subconscious-daemon/*.py prompts/*.txt | tail -1
