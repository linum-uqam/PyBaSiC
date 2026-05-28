(contributing)=
# Contributing

Thank you for contributing to Linum BaSiC!  This page covers how to set up a
development environment, run the tests and linters, and the conventions we
follow.

---

## Development environment

Linum BaSiC uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/linum-uqam/Linum-BaSiC.git
cd Linum-BaSiC

# Create a virtual environment and install dev dependencies
make install
# or equivalently: uv sync --extra dev
```

For GPU support:

```bash
make install-gpu
# or: uv sync --extra dev --extra gpu
```

---

## Make targets

| Target | Command | Description |
|---|---|---|
| `install` | `uv sync --extra dev` | Install dev dependencies |
| `install-gpu` | `uv sync --extra dev --extra gpu` | Install dev + GPU deps |
| `lint` | `ruff check` | Run Ruff linter |
| `format` | `ruff format` | Format code in-place |
| `format-check` | `ruff format --check` | Check formatting without writing |
| `typecheck` | `ty check linum_basic` | Run `ty` type checker |
| `test` | `pytest -q` | Run test suite |
| `all` | lint + format-check + typecheck + test | Full CI check locally |
| `docs` | `sphinx-build -W --keep-going -n -b html docs docs/_build/html` | Build docs |
| `docs-live` | `sphinx-autobuild docs docs/_build/html` | Live-reload docs server |

---

## Running the tests

```bash
make test
# or: uv run pytest -q
```

For verbose output and to see which tests are skipped:

```bash
uv run pytest -v --tb=short
```

---

## Building the docs locally

```bash
# Install docs dependencies
uv sync --extra docs

# Build HTML docs (warnings treated as errors)
make docs

# Open in browser
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
```

For live-reloading during writing:

```bash
make docs-live
# then open http://127.0.0.1:8000
```

---

## Pre-commit

We recommend using [pre-commit](https://pre-commit.com/) to catch
formatting and lint issues before committing.  It is already included in the
`dev` extras, so after `make install` you can activate it with:

```bash
# If you installed dev dependencies with uv (recommended)
uv run pre-commit install

# Or with pip
pip install pre-commit
pre-commit install
```

---

## Docstring conventions

All docstrings follow **NumPy style** as configured in `pyproject.toml`
(`tool.ruff.lint.pydocstyle.convention = "numpy"`).

Example:

```python
def my_function(x: float, y: float) -> float:
    """One-line summary.

    Extended description of what the function does, optional.

    Parameters
    ----------
    x : float
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    float
        Description of the return value.

    Raises
    ------
    ValueError
        If y is zero.

    Examples
    --------
    >>> my_function(1.0, 2.0)
    0.5
    """
    if y == 0:
        raise ValueError("y must not be zero")
    return x / y
```

Key rules:

- First line: short imperative summary, no trailing period.
- Sections: `Parameters`, `Returns`, `Raises`, `Notes`, `References`,
  `Examples` (in that order, omit if empty).
- Type annotations go in the function signature, **not** repeated in the
  docstring.

---

## Code style

- **Python ≥ 3.14**, type annotations required for all public functions.
- **Ruff** for linting and formatting (configured in `pyproject.toml`).
- **No new dependencies** without discussion — Linum BaSiC has a deliberately
  lean dependency set.
- **Surgical changes** — PRs should touch only what is needed.

---

## Submitting a pull request

1. Fork the repository and create a feature branch.
2. Write tests for any new behaviour.
3. Run `make all` locally and ensure it passes.
4. Open a PR against `master`.  The CI will run linting and tests.
