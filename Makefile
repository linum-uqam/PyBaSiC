.PHONY: install install-gpu lint format typecheck test test-gpu-smoke all docs docs-live

install:
	uv sync --extra dev

install-gpu:
	uv sync --extra dev --extra gpu

lint:
	uv run ruff check

format:
	uv run ruff format

format-check:
	uv run ruff format --check

typecheck:
	uv run ty check linum_basic/

test:
	uv run python -m pytest -q --ignore=tests/test_demo_fitting.py

test-gpu-smoke:
	bash scripts/gpu_smoke.sh

all: lint format-check typecheck test

docs:
	uv run sphinx-build -W --keep-going -n -b html docs docs/_build/html

docs-live:
	uv run sphinx-autobuild docs docs/_build/html
