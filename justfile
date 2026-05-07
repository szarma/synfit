# Quality gate commands for synfit

# Default recipe: full quality gate
default:
    @just check

# Full quality gate: install + all tests
check:
    @echo "Running full quality gate..."
    @just install
    @just test
    @echo "Quality gate passed!"

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

# Show available commands
help:
    @just --list
