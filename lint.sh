#!/usr/bin/env bash
set -e

echo "=== Running Ruff Linter & Formatter ==="
uv run ruff check --fix .
uv run ruff format .
echo "=== Linter & Formatting Checks Passed! ==="
