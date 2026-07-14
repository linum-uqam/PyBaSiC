"""Docstring coverage tests using numpydoc.validate.

Validates that every public object in the ``linum_basic`` package has a
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
# GL01 = Summary on same line as opening quotes — conflicts with ruff D212
# GL08 = Object does not have a docstring  (caught separately)
# SA01 = See Also section not found
# EX01 = No examples section
# ES01 = No extended summary — optional in NumPy style
# PR01 = Parameters not documented — suppressed for inherited/internal params
_ALLOWED_CODES: frozenset[str] = frozenset(
    {
        "GL01",  # ruff D212 (numpy convention) requires summary on first line
        "SA01",  # "See Also" is optional
        "EX01",  # Examples section is optional for private helpers
        "ES01",  # Extended summary is optional
        "PR01",  # Inherited / StrEnum internal parameters
    }
)

# Objects whose codes should be entirely ignored (private / test helpers).
_SKIP_OBJECTS: frozenset[str] = frozenset(
    {
        "linum_basic.backend._torch_dct1d",
        "linum_basic.backend._torch_idct1d",
        "linum_basic.backend._torch_dctn",
        "linum_basic.backend._torch_idctn",
        "linum_basic.backend.ArrayNamespace._numpy_dtype_to_torch",
        "linum_basic.viz.Panel.__init__",
        "linum_basic.benchmark.strategies.WorkloadContext.__init__",
        "linum_basic.benchmark.strategies.AutoStrategyResult.__init__",
        "linum_basic._working_size.WorkingSizeContext.__init__",
        "linum_basic._working_size.WorkingSizeResolution.__init__",
        "linum_basic.benchmark.profile.HistoricalBaseline.__init__",
        "linum_basic.benchmark.audit.TraceabilityRow.__init__",
        "linum_basic.benchmark.audit.AuditFinding.__init__",
        "linum_basic.benchmark.audit.MicroBenchmarkResult.__init__",
        "linum_basic.benchmark.audit.HotspotRecord.__init__",
    }
)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _iter_public_objects() -> list[tuple[str, object]]:
    """Walk ``linum_basic`` and yield ``(qualified_name, object)`` for every public
    function, method, and class.

    Returns
    -------
    list of tuple
        Each item is ``(qualified_name, callable_or_class)``.
    """
    import linum_basic

    result: list[tuple[str, object]] = []
    seen: set[int] = set()

    def _visit(mod: ModuleType, prefix: str) -> None:
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            qname = f"{prefix}.{name}"
            if id(obj) in seen:
                continue
            # Skip objects not defined in this module (e.g. stdlib imports).
            if getattr(obj, "__module__", None) != mod.__name__:
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

    _visit(linum_basic, "linum_basic")
    for _info in pkgutil.walk_packages(linum_basic.__path__, prefix="linum_basic."):
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
    errors = [(code, msg) for code, msg in result.get("errors", []) if code not in _ALLOWED_CODES]
    if errors:
        formatted = "\n".join(f"  [{code}] {msg}" for code, msg in errors)
        pytest.fail(f"Docstring issues in {qname}:\n{formatted}")
