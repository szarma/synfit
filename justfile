# Quality gate commands for synfit

# Default recipe: full quality gate
default:
    @just check

# Full quality gate: install + all tests
check:
    @echo "Running full quality gate..."
    @just install
    @just check-readme-examples
    @just test
    @echo "Quality gate passed!"

# Execute README and tutorial Python blocks against the current environment
check-readme-examples:
    uv run python scripts/check_readme_examples.py

# Regenerate the figures from the exact snippets shown in the documentation
generate-doc-figures:
    uv run python scripts/generate_doc_figures.py

# Generate the PyPI long description with version-pinned public image URLs
generate-pypi-readme:
    uv run python scripts/generate_doc_figures.py --generate-pypi-readme

# Install dependencies
install:
    uv sync --extra dev

# Run all tests
test:
    uv run pytest -v --tb=short

# Run tests with coverage
test-coverage:
    uv run pytest --cov=src --cov-report=html -v --tb=short
    @echo "Coverage report: htmlcov/index.html"

# Generate committed synthetic seed CSVs from config.json files
generate-seed-data:
    uv run python scripts/generate_seed_datasets.py

# Verify seed CSVs are up-to-date with configs (CI check)
check-seed-data:
    uv run python scripts/generate_seed_datasets.py --check

# Regenerate tests/data/response_baseline.json after intentional output changes
update-response-baseline:
    uv run python scripts/update_response_baseline.py --write

# Show available commands
help:
    @just --list
