"""Contract tests for the M007/S01 streaming-vs-eager peak-memory probe.

These tests prove the probe produces a *well-formed, honest* JSON artifact
(schema, non-negative peaks, correct mode wiring) against a small synthetic
OME-Zarr volume on any host (including CI on macOS without a GPU). They do
*not* attempt to prove the production-scale memory delta — that requires the
real A6000 volume and is the job of T02/T03, which invoke this script twice as
independent processes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import streaming_memory_probe as probe
from tests.test_cli import _write_synthetic_mosaic_zarr

# Small, fast BaSiC kwargs for contract tests (numpy backend, no CUDA). The
# explicit ``backend='numpy'`` keeps the contract tests runnable on any host
# (CI on macOS without CUDA); the real eager/streaming A/B on the A6000 is
# the job of T02/T03, which pass ``--backend torch --device cuda:0``.
_FIT_KW = {"working_size": 16, "max_reweighting_iterations": 3, "estimate_darkfield": False, "backend": "numpy"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_zarr(tmp_path: Path) -> Path:
    """A tiny real OME-Zarr mosaic volume the probe can fit."""
    zarr_path = tmp_path / "subject.ome.zarr"
    _write_synthetic_mosaic_zarr(zarr_path, n_z=4, n_rows=3, n_cols=3, tile=16)
    return zarr_path


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

# Every key the S01 verification contract demands, mapped to its expected JSON
# type. ``peak_vram_bytes`` is nullable (None on hosts without CUDA) so it is
# allowed to be int or None.
_REQUIRED_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "mode": str,
    "peak_rss_bytes": int,
    "peak_vram_bytes": (int, type(None)),
    "wall_ms": (int, float),
    "input_path": str,
    "z_indices": list,
    "n_z": int,
    "working_size": int,
    "backend": (str, type(None)),
    "device": (str, type(None)),
    "git_commit": str,
    "host": dict,
}


class TestArtifactSchema:
    """The JSON artifact must carry every contract field with the right type."""

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_artifact_has_required_schema(self, synthetic_zarr: Path, mode: str) -> None:
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)

        for key, expected_type in _REQUIRED_SCHEMA.items():
            assert key in artifact, f"artifact missing required key {key!r}"
            assert isinstance(artifact[key], expected_type), f"artifact[{key!r}]={artifact[key]!r} not {expected_type!r}"

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_mode_is_recorded_honestly(self, synthetic_zarr: Path, mode: str) -> None:
        """The recorded mode/streaming/lazy_load must match what was requested."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        assert artifact["mode"] == mode
        assert artifact["streaming"] is (mode == "streaming")
        assert artifact["lazy_load"] is (mode == "streaming")

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_peak_values_are_non_negative(self, synthetic_zarr: Path, mode: str) -> None:
        """Peak RSS is always a positive int; VRAM is None (no CUDA) or non-negative."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        assert artifact["peak_rss_bytes"] > 0
        if artifact["peak_vram_bytes"] is not None:
            assert artifact["peak_vram_bytes"] >= 0

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_wall_ms_is_positive(self, synthetic_zarr: Path, mode: str) -> None:
        """A real fit must take measurable time."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        assert artifact["wall_ms"] > 0

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_z_indices_match_volume(self, synthetic_zarr: Path, mode: str) -> None:
        """z_indices must be valid indices into the volume and respect the request."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=3, **_FIT_KW)
        zs = artifact["z_indices"]
        assert len(zs) == 3
        assert all(0 <= z < artifact["n_z"] for z in zs)
        assert zs == sorted(set(zs))

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_host_platform_recorded(self, synthetic_zarr: Path, mode: str) -> None:
        """The host block must carry platform + python version for cross-host audit."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        host = artifact["host"]
        assert "platform" in host and isinstance(host["platform"], str) and host["platform"]
        assert host.get("python_version")


# ---------------------------------------------------------------------------
# Numerics sanity: both modes produce a real BaSiC fit
# ---------------------------------------------------------------------------


class TestProbeProducesFit:
    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_flatfields_shape_matches_z_selection(self, synthetic_zarr: Path, mode: str) -> None:
        """The probe actually ran a fit: flatfields have one plane per z."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        assert artifact["flatfields_shape"][0] == 2

    @pytest.mark.parametrize("mode", ["eager", "streaming"])
    def test_artifact_is_json_serialisable(self, synthetic_zarr: Path, mode: str, tmp_path: Path) -> None:
        """The artifact round-trips through JSON (T02/T03 persist it to disk)."""
        artifact = probe.run_probe(synthetic_zarr, mode=mode, z_sample=2, **_FIT_KW)
        out = tmp_path / "probe.json"
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded["mode"] == mode
        assert reloaded["peak_rss_bytes"] == artifact["peak_rss_bytes"]


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestProbeNegativePaths:
    def test_unknown_mode_raises(self, synthetic_zarr: Path) -> None:
        with pytest.raises(ValueError, match="Unknown mode"):
            probe.run_probe(synthetic_zarr, mode="bogus", z_sample=2, **_FIT_KW)

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.ome.zarr"
        with pytest.raises(FileNotFoundError, match="Input not found"):
            probe.run_probe(missing, mode="eager", z_sample=2, **_FIT_KW)

    def test_no_z_selection_raises(self, synthetic_zarr: Path) -> None:
        """resolve_z_selection requires explicit z-selection (reused, not duplicated)."""
        with pytest.raises(ValueError, match=r"z-indices|z-sample"):
            probe.run_probe(synthetic_zarr, mode="eager", **_FIT_KW)

    def test_both_z_selections_raises(self, synthetic_zarr: Path) -> None:
        with pytest.raises(ValueError, match="only one"):
            probe.run_probe(synthetic_zarr, mode="eager", z_indices="0", z_sample=2, **_FIT_KW)

    def test_large_run_guard_without_flag_raises_systemexit(self, tmp_path: Path) -> None:
        """Selecting more z-planes than the threshold refuses without --yes."""
        zarr_path = tmp_path / "big.ome.zarr"
        # n_z just above the LARGE_RUN_Z_THRESHOLD so z_sample selects all of them.
        _write_synthetic_mosaic_zarr(zarr_path, n_z=probe.LARGE_RUN_Z_THRESHOLD + 2, n_rows=2, n_cols=2, tile=16)
        with pytest.raises(SystemExit):
            probe.run_probe(
                zarr_path,
                mode="eager",
                z_sample=probe.LARGE_RUN_Z_THRESHOLD + 2,
                allow_large_run=False,
                yes=False,
                **_FIT_KW,
            )

    def test_large_run_guard_yes_flag_allows(self, tmp_path: Path) -> None:
        """The --yes / --allow-large-run flag lifts the guard."""
        zarr_path = tmp_path / "big.ome.zarr"
        _write_synthetic_mosaic_zarr(zarr_path, n_z=probe.LARGE_RUN_Z_THRESHOLD + 2, n_rows=2, n_cols=2, tile=16)
        artifact = probe.run_probe(
            zarr_path,
            mode="streaming",
            z_sample=probe.LARGE_RUN_Z_THRESHOLD + 2,
            allow_large_run=False,
            yes=True,
            **_FIT_KW,
        )
        assert artifact["mode"] == "streaming"

    def test_z_index_out_of_range_raises(self, synthetic_zarr: Path) -> None:
        with pytest.raises(ValueError, match="out of range"):
            probe.run_probe(synthetic_zarr, mode="eager", z_indices="999", **_FIT_KW)


# ---------------------------------------------------------------------------
# CLI surface (main writes the JSON artifact and returns 0)
# ---------------------------------------------------------------------------


class TestProbeCLI:
    def test_main_writes_artifact_and_returns_zero(self, synthetic_zarr: Path, tmp_path: Path) -> None:
        out = tmp_path / "out" / "probe.json"
        rc = probe.main(
            [
                "--input",
                str(synthetic_zarr),
                "--mode",
                "eager",
                "--output",
                str(out),
                "--z-sample",
                "2",
                "--backend",
                "numpy",
                "--working-size",
                "16",
            ]
        )
        assert rc == 0
        assert out.exists()
        artifact = json.loads(out.read_text(encoding="utf-8"))
        assert artifact["mode"] == "eager"
        assert artifact["peak_rss_bytes"] > 0

    def test_main_missing_input_returns_one(self, tmp_path: Path) -> None:
        out = tmp_path / "probe.json"
        rc = probe.main(
            ["--input", str(tmp_path / "nope.ome.zarr"), "--mode", "eager", "--output", str(out), "--z-sample", "2"]
        )
        assert rc == 1
        assert not out.exists()

    def test_argparse_requires_mode_and_z_selection(self, synthetic_zarr: Path, tmp_path: Path) -> None:
        """Missing required args exit with code 2 (argparse error)."""
        with pytest.raises(SystemExit) as excinfo:
            probe.main(["--input", str(synthetic_zarr), "--output", str(tmp_path / "p.json")])
        assert excinfo.value.code == 2
