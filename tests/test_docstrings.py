"""Docstring coverage tests using numpydoc.validate.

Validates that every public object in the ``pybasic`` package has a
well-formed NumPy-style docstring.  A test is generated for each object so
that failures are reported individually rather than as a single bulk error.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import FunctionType, ModuleType

import pytest

numpydoc_validate = pytest.importorskip(
    "numpydoc.validate",
    reason="numpydoc not installed — install dev extras: uv sync --extra dev",
)

# ---------------------------------------------------------------------------
# Allowed warning codes that do not constitute failures.
# ---------------------------------------------------------------------------
# GL08 = Object does not have a docstring  (caught separately)
# SA01 = See Also section not found
# EX01 = No examples section
_ALLOWED_CODES: frozenset[str] = frozenset(
    {
        "SA01",  # "See Also" is optional
        "EX01",  # Examples section is optional for private helpers
    }
)

# Objects whose codes should be entirely ignored (private / test helpers).
_SKIP_OBJECTS: frozenset[str] = frozenset(
    {
        "pybasic.backend._torch_dct1d",
        "pybasic.backend._torch_idct1d",
        "pybasic.backend._torch_dctn",
        "pybasic.backend._torch_idctn",
        "pybasic.backend.ArrayNamespace._numpy_dtype_to_torch",
    }
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _iter_public_objects() -> list[tuple[str, object]]:
    """Walk ``pybasic`` and yield ``(qualified_name, object)`` for every public
    function, method, and class.

    Returns
    -------
    list of tuple
        Each item is ``(qualified_name, callable_or_class)``.
    """
    import pybasic  # noqa: PLC0415

    result: list[tuple[str, object]] = []
    seen: set[int] = set()

    def _visit(mod: ModuleType, prefix: str) -> None:
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            qname = f"{prefix}.{name}"
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.isclass(obj):
                result.append((qname, obj))
                for mname, mobj in inspect.getmembers(obj, predicate=inspect.isfunction):
                    if mname.startswith("__") and mname != "__init__":
                        continue
                    if mname.startswith("_") and not mname.startswith("__"):
                        continue
                    result.append((f"{qname}.{mname}", mobj))
            elif isinstance(obj, FunctionType):
                result.append((qname, obj))

    _visit(pybasic, "pybasic")
    for _info in pkgutil.walk_packages(pybasic.__path__, prefix="pybasic."):
        try:
            mod = importlib.import_module(_info.name)
        except ImportError:
            continue
        _visit(mod, _info.name)

    return result


# Build the parameter list once at collection time.
_PUBLIC_OBJECTS = _iter_public_objects()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qname,obj",
    _PUBLIC_OBJECTS,
    ids=[qname for qname, _ in _PUBLIC_OBJECTS],
)
def test_docstring_valid(qname: str, obj: object) -> None:
    """Each public object passes numpydoc validation.

    Parameters
    ----------
    qname : str
        Fully qualified name of the object under test.
    obj : object
        The callable or class to validate.
    """
    if qname in _SKIP_OBJECTS:
        pytest.skip(f"Skipping private/internal object: {qname}")

    doc = inspect.getdoc(obj)
    if not doc:
        pytest.fail(f"{qname} has no docstring (GL08).")

    result = numpydoc_validate.validate(f"{qname}")
    errors = [
        (code, msg)
        for code, msg in result.get("errors", [])
        if code not in _ALLOWED_CODES
    ]
    if errors:
        formatted = "\n".join(f"  [{code}] {msg}" for code, msg in errors)
        pytest.fail(f"Docstring issues in {qname}:\n{formatted}")
