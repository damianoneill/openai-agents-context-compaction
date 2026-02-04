.PHONY: help sync lint format typecheck test coverage check pre-commit-install pre-commit clean bump bump-minor bump-major changelog release

# Default target
.DEFAULT_GOAL := help

# Show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  sync            Install/update dependencies"
	@echo "  format          Format code with ruff"
	@echo "  lint            Run linter"
	@echo "  typecheck       Run type checker (mypy)"
	@echo "  test            Run tests"
	@echo "  coverage        Run tests with coverage (90% minimum)"
	@echo "  check           Run all checks (format, lint, typecheck, coverage)"
	@echo ""
	@echo "Git Hooks:"
	@echo "  pre-commit-install  Install pre-commit hooks"
	@echo "  pre-commit          Run pre-commit on all files"
	@echo ""
	@echo "Release:"
	@echo "  bump            Auto-bump version based on commits"
	@echo "  bump-minor      Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  bump-major      Bump major version (0.1.0 -> 1.0.0)"
	@echo "  changelog       Generate/update changelog"
	@echo "  release         Run checks + bump + changelog + tag"
	@echo ""
	@echo "Utilities:"
	@echo "  clean           Remove build artifacts"

# Sync dependencies (install + lock)
sync:
	uv sync --all-extras

# Run linter
lint:
	uv run ruff check src tests

# Run formatter
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Run type checker
typecheck:
	uv run mypy src --python-version 3.11

# Run tests
test:
	uv run pytest -v

# Run tests with coverage (enforce minimum 90%)
coverage:
	uv run pytest --cov=src/openai_agents_context_compaction --cov-report=term-missing --cov-fail-under=90

# Run all checks (format, lint, typecheck, tests with coverage)
check: format lint typecheck coverage

# Install pre-commit hooks
pre-commit-install:
	uv run pre-commit install

# Run pre-commit on all files
pre-commit:
	uv run pre-commit run --all-files

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Version bumping with commitizen (auto-detects bump type from commits)
bump:
	uv run cz bump

# Bump minor version (e.g., 0.1.0 -> 0.2.0)
bump-minor:
	uv run cz bump --increment MINOR

# Bump major version (e.g., 0.1.0 -> 1.0.0)
bump-major:
	uv run cz bump --increment MAJOR

# Generate/update changelog from commits
changelog:
	uv run cz changelog

# Full release: run checks, bump version, update changelog, create tag
release: check
	uv run cz bump --changelog
