"""Tests for the package-level public API export contract.

Guards the top-level re-exports declared in ``linum_basic/__init__.py`` so an
accidental removal of a symbol from the import block or ``__all__`` fails fast.
Focuses on the symbols added in M008/S02 (``auto_tune`` family) alongside the
existing tuning re-exports they mirror.
"""

from __future__ import annotations

import linum_basic
import linum_basic.tuning as tuning

# Symbols the S02 design contract (docs/auto_apply_safety_gate.md) designates as
# the public auto-apply entry point, re-exported at the package top level.
AUTO_APPLY_SYMBOLS = ["auto_tune", "AutoTuneResult", "AutoApplyError"]

# Pre-existing tuning re-exports that the new symbols must sit alongside.
EXISTING_TUNING_SYMBOLS = ["tune", "recommend_bounds", "TuneResult", "BoundsRecommendation"]


class TestPublicApiExports:
    """The new auto-apply entry point must be importable from the package top."""

    def test_auto_apply_symbols_importable_from_top_level(self):
        import linum_basic as lb

        for name in AUTO_APPLY_SYMBOLS:
            assert hasattr(lb, name), f"linum_basic is missing top-level export {name!r}"

    def test_auto_apply_symbols_are_same_objects_as_submodule(self):
        """Top-level names must be the exact objects from linum_basic.tuning."""
        for name in AUTO_APPLY_SYMBOLS:
            top = getattr(linum_basic, name)
            sub = getattr(tuning, name)
            assert top is sub, (
                f"linum_basic.{name} must be the same object as linum_basic.tuning.{name}, got {top!r} vs {sub!r}"
            )

    def test_auto_apply_symbols_listed_in_dunder_all(self):
        for name in AUTO_APPLY_SYMBOLS:
            assert name in linum_basic.__all__, f"{name!r} must be in linum_basic.__all__"

    def test_existing_tuning_reexports_still_present(self):
        """Adding the new symbols must not displace the existing tuning API."""
        for name in EXISTING_TUNING_SYMBOLS:
            assert hasattr(linum_basic, name), f"existing export {name!r} was lost"
            assert name in linum_basic.__all__, f"existing export {name!r} dropped from __all__"

    def test_dunder_all_entries_resolve(self):
        """Every entry in __all__ must resolve to a real attribute (no dangling refs)."""
        missing = [name for name in linum_basic.__all__ if not hasattr(linum_basic, name)]
        assert not missing, f"__all__ lists unresolvable names: {missing}"
