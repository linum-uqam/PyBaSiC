.PHONY: install install-gpu lint format typecheck test all

install:
	uv sync --extra dev

install-gpu:
	uv sync --extra dev --extra gpu

lint:
	uv run ruff check pybasic tests

format:
	uv run ruff format pybasic tests

format-check:
	uv run ruff format --check pybasic tests

typecheck:
	uv run ty check pybasic

test:
	uv run pytest -q

all: lint format-check typecheck test
