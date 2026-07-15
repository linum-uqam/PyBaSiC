r"""Command-line interface for BaSiC shading correction.

Entry point registered as ``basic`` in *pyproject.toml*.  All sub-tasks are
exposed as sub-commands of the same executable:

Usage
-----
::

    basic correct  --input /path/to/tiles --output /path/to/output
    basic fit      --input mosaic.ome.zarr --output corrected.ome.zarr
    basic tune     --input mosaic.ome.zarr
    basic preview  --input volume.ome.zarr --output preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linum_basic.mosaic import MosaicGrid

# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------


def _validate_device(device: str | None) -> int | None:
    """Reject unsupported MPS device strings before backend dispatch."""
    if device and device.lower().startswith("mps"):
        print(
            "error: MPS is not supported for BaSiC fitting; use --backend numpy or --device cuda:0.",
            file=sys.stderr,
        )
        return 1
    return None


def _working_size_type(value: str) -> int | str:
    """Parse a ``--working-size`` value into an int or the ``"auto"`` sentinel."""
    if value == "auto":
        return "auto"
    try:
        ws = int(value)
    except TypeError, ValueError:
        msg = f"working_size must be an integer or 'auto', got {value!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if ws <= 0:
        msg = f"working_size must be positive, got {ws}"
        raise argparse.ArgumentTypeError(msg) from None
    return ws


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--backend`` and ``--device`` to *parser*."""
    g = parser.add_argument_group("Compute")
    g.add_argument(
        "--backend",
        choices=["numpy", "torch", "auto"],
        default="numpy",
        help="Array backend for the ALM optimisation loop.  'auto' selects Torch with CUDA when available, else NumPy.",
    )
    g.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="PyTorch device string (e.g. 'cuda:0', 'cpu').  Ignored when --backend=numpy.",
    )


# ---------------------------------------------------------------------------
# Sub-command: correct
# ---------------------------------------------------------------------------


def _add_correct_subcommand(subs: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subs.add_parser(
        "correct",
        help="Estimate and apply BaSiC flat-field/dark-field shading correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", metavar="DIR", required=True, type=Path, help="Directory containing the input image stack.")
    io.add_argument("--output", metavar="DIR", required=True, type=Path, help="Directory to write corrected images.")
    io.add_argument("--extension", metavar="EXT", default=".tif", help="File extension filter (e.g. '.tif', '.png').")

    alg = p.add_argument_group("Algorithm")
    alg.add_argument(
        "--estimate-darkfield",
        action="store_true",
        default=False,
        help="Estimate the dark-field in addition to the flat-field.",
    )

    _add_backend_args(p)
    p.add_argument("--verbose", action="store_true", default=False, help="Print progress bars and iteration statistics.")


def _run_correct(args: argparse.Namespace) -> int:
    if (rc := _validate_device(args.device)) is not None:
        return rc

    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"error: --input '{input_dir}' is not an existing directory.", file=sys.stderr)
        return 1

    from linum_basic.core import BaSiC

    model = BaSiC(
        input_dir,
        estimate_darkfield=args.estimate_darkfield,
        extension=args.extension,
        verbose=args.verbose,
        backend=args.backend,
        device=args.device,
    )
    model.run()  # auto-calls prepare()
    model.write_images(output_dir)

    if args.verbose:
        print(f"Saved corrected images to '{output_dir}'.")

    return 0


# ---------------------------------------------------------------------------
# Sub-command: fit
# ---------------------------------------------------------------------------


def _add_fit_subcommand(subs: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subs.add_parser(
        "fit",
        help="Fit BaSiC flat/dark-fields on an OME-Zarr mosaic grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr mosaic.")
    io.add_argument("--output", metavar="ZARR", required=True, type=Path, help="Path to write the corrected .ome.zarr mosaic.")
    io.add_argument(
        "--save-fields", metavar="DIR", default=None, type=Path, help="Directory to save flat/dark-field arrays as .npy files."
    )

    alg = p.add_argument_group("Algorithm")
    alg.add_argument("--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    alg.add_argument(
        "--estimate-darkfield",
        action="store_true",
        default=False,
        help="Estimate the dark-field in addition to the flat-field.",
    )
    alg.add_argument(
        "--field-mode",
        choices=["per-z", "global"],
        default="per-z",
        help="'per-z': fit one field per z-level. 'global': average all per-z fields.",
    )
    alg.add_argument(
        "--working-size",
        metavar="WS|auto",
        default=None,
        type=_working_size_type,
        help=(
            "BaSiC internal resolution (default 128, left to BaSiC when omitted). "
            "Pass 'auto' (opt-in) to resolve a grid size from memory budget and "
            "preview quality; the decision is recorded in the fit metadata."
        ),
    )
    alg.add_argument("--z-indices", metavar="Z", nargs="+", type=int, default=None, help="Z-levels to fit (default: all).")
    alg.add_argument(
        "--strategy",
        choices=["auto", "sequential", "multi", "batched"],
        default="auto",
        help=(
            "Execution strategy. 'auto' selects the path from workload shape and GPU count; "
            "when auto picks a CUDA path the effective backend becomes torch unless --backend is set."
        ),
    )
    alg.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help=(
            "Fit z-levels one at a time (sequential only), bounding peak memory to a single "
            "plane instead of holding all per-z tile stacks live. Incompatible with "
            "--strategy multi or batched."
        ),
    )
    alg.add_argument(
        "--lazy",
        action="store_true",
        default=False,
        help=(
            "Load the input mosaic lazily (zarr.Array-backed) so the full volume is not read "
            "into memory up front. Pair with --streaming for the lowest peak memory on the "
            "fit path."
        ),
    )

    # Fit omits backend/device from basic_kwargs unless explicitly passed (default None),
    # so strategy=auto can upgrade to torch for CUDA paths without clobbering by numpy default.
    compute = p.add_argument_group("Compute")
    compute.add_argument(
        "--backend",
        choices=["numpy", "torch", "auto"],
        default=None,
        help="Array backend for the ALM loop.  Omit to let strategy=auto choose (may select torch for CUDA).",
    )
    compute.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="PyTorch device string (e.g. 'cuda:0', 'cpu').  Ignored when --backend=numpy.",
    )
    p.add_argument(
        "--n-jobs",
        metavar="N",
        type=int,
        default=None,
        help="Worker processes for parallel z-level fitting (default: CPU count - 2).",
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Show progress bars.")


def _run_fit(args: argparse.Namespace) -> int:
    if (rc := _validate_device(args.device)) is not None:
        return rc

    from linum_basic.fit import fit_mosaic, save_corrected
    from linum_basic.mosaic import MosaicGrid

    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap, lazy=args.lazy)
    basic_kwargs: dict = {"estimate_darkfield": args.estimate_darkfield}
    if args.backend is not None:
        basic_kwargs["backend"] = args.backend
    if args.device is not None:
        basic_kwargs["device"] = args.device
    if args.working_size is not None:
        basic_kwargs["working_size"] = args.working_size
    fit = fit_mosaic(
        mosaic,
        z_indices=args.z_indices,
        field_mode=args.field_mode,
        basic_kwargs=basic_kwargs,
        strategy=args.strategy,
        streaming=args.streaming,
        n_workers=args.n_jobs,
        verbose=args.verbose,
    )

    save_corrected(mosaic, fit, args.output, input_path=args.input, overwrite=True)

    if args.save_fields:
        import numpy as np

        args.save_fields.mkdir(parents=True, exist_ok=True)
        np.save(args.save_fields / "flatfields.npy", fit.flatfields)
        np.save(args.save_fields / "darkfields.npy", fit.darkfields)
        if args.verbose:
            print(f"Saved fields to '{args.save_fields}'.")

    if args.verbose:
        strategy_meta = fit.params.get("_strategy", {})
        execution_path = strategy_meta.get("execution_path")
        reason_summary = strategy_meta.get("reason_summary")
        if execution_path is not None:
            print(f"Strategy execution path: {execution_path}")
        if reason_summary:
            print(f"Strategy reason: {reason_summary}")
        print(f"Saved corrected mosaic to '{args.output}'.")

    return 0


# ---------------------------------------------------------------------------
# Sub-command: tune
# ---------------------------------------------------------------------------


def _add_tune_subcommand(subs: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subs.add_parser(
        "tune",
        help="Tune BaSiC hyperparameters on an OME-Zarr mosaic using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr mosaic.")
    io.add_argument(
        "--out-json", metavar="FILE", default=None, type=Path, help="Write best hyperparameters as JSON to this file."
    )
    io.add_argument(
        "--apply", metavar="ZARR", default=None, type=Path, help="If set, run a full-z fit with the best params and save here."
    )
    io.add_argument(
        "--bounds-json",
        metavar="FILE",
        default=None,
        type=Path,
        help="Write the recommended narrowed search-space bounds (recommend_bounds) as JSON to this file.",
    )
    io.add_argument(
        "--auto-apply",
        action="store_true",
        default=False,
        help=(
            "Run the auto-apply safety gate: fit one default-bounds baseline and one "
            "narrowed-bounds candidate, gate the candidate on full-volume seam_l1 + "
            "seam_curvature deltas (fixed 0.0 non-regression margin), and write the "
            "winning fit to --apply (or report the fallback). Independent of "
            "--bounds-json / --bounds-margin / --out-json / --apply."
        ),
    )

    tuning = p.add_argument_group("Tuning")
    tuning.add_argument("--n-trials", metavar="N", type=int, default=50, help="Number of Optuna trials.")
    tuning.add_argument("--z-subsample", metavar="N", type=int, default=4, help="Z-levels evaluated per trial.")
    tuning.add_argument("--storage", metavar="URL", default=None, help="Optuna storage URL, e.g. sqlite:///tune.db.")
    tuning.add_argument(
        "--study-name", metavar="NAME", default="basic-tune", help="Optuna study name (for persistent storage)."
    )
    tuning.add_argument("--seed", metavar="N", type=int, default=0, help="Random seed for reproducibility.")
    tuning.add_argument(
        "--n-jobs", metavar="N", type=int, default=None, help="Worker threads for z-level evaluation within each trial."
    )
    tuning.add_argument(
        "--max-tiles",
        metavar="N",
        type=int,
        default=64,
        help="Tiles (evenly spaced) used for the seam metric per trial. Use 0 for all tiles.",
    )
    tuning.add_argument(
        "--n-extra-rows",
        metavar="N",
        type=int,
        default=0,
        help="Leading rows per tile to drop before fitting (galvo fly-back artefact).",
    )
    tuning.add_argument("--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    tuning.add_argument(
        "--working-size",
        metavar="WS|auto",
        default=128,
        type=_working_size_type,
        help=(
            "Controls the working_size search dimension during tuning (default 128: "
            "Optuna explores the full grid). Pass 'auto' (opt-in) to resolve a single "
            "grid size from memory budget and preview quality before the search."
        ),
    )
    tuning.add_argument(
        "--bounds-margin",
        metavar="FRAC",
        type=float,
        default=0.10,
        help="Relative margin (0-1) selecting the near-optimal trial band used by --bounds-json.",
    )

    backend = p.add_argument_group("Backend")
    backend.add_argument(
        "--backend", metavar="NAME", default="numpy", choices=["numpy", "torch"], help="Compute backend ('numpy' or 'torch')."
    )
    backend.add_argument(
        "--device", metavar="DEV", default=None, help="Torch device string, e.g. 'cuda:0'. Ignored when --backend=numpy."
    )

    p.add_argument("--verbose", action="store_true", default=False, help="Enable Optuna logging and progress bars.")


def _run_auto_apply(args: argparse.Namespace, mosaic: MosaicGrid) -> int:
    """Run the D023 auto-apply safety gate via :func:`auto_tune` and write outputs.

    Delegates the full pipeline (baseline fit + tune + recommend_bounds +
    candidate fit + non-regression gate) to :func:`auto_tune`, then composes
    with the existing output flags: ``--out-json`` writes the candidate best
    params, ``--bounds-json`` writes the narrowed recommendation, and
    ``--apply`` writes the *winning* fit (candidate on ``pass``, baseline on
    fallback). ``--auto-apply`` is independent of those flags but composes
    with all of them.
    """
    from linum_basic.tuning import AutoApplyError, auto_tune

    try:
        at = auto_tune(
            mosaic,
            n_trials=args.n_trials,
            z_subsample=args.z_subsample,
            margin=args.bounds_margin,
            seed=args.seed,
            n_workers=args.n_jobs,
            backend=args.backend,
            device=args.device,
            max_tiles=args.max_tiles if args.max_tiles > 0 else None,
            n_extra_rows=args.n_extra_rows,
            verbose=args.verbose,
        )
    except AutoApplyError as exc:
        print(f"error: auto-apply failed: {exc}", file=sys.stderr)
        return 1

    gate = at.gate

    if args.verbose:
        print(f"Auto-apply gate verdict: {gate['gate_verdict']}")
        print(f"  applied: {gate['applied']}")
        if gate.get("failing_metrics"):
            print(f"  failing_metrics: {', '.join(gate['failing_metrics'])}")
        if gate.get("fallback_reason") is not None:
            print(f"  fallback_reason: {gate['fallback_reason']}")
        deltas = gate.get("deltas") or {}
        for metric in ("seam_l1", "seam_curvature"):
            if metric in deltas:
                d = deltas[metric]
                print(f"  {metric}: abs_delta={d['abs_delta']:.6f} rel_delta={d['rel_delta']:.6f}")

    # ``--out-json``: write the candidate best params when available. On a
    # fallback that fired before a candidate existed (e.g. tune-failed) the
    # candidate sub-dict is empty, so skip the write.
    candidate_meta = gate.get("candidate") or {}
    if args.out_json and candidate_meta.get("best_params"):
        import json

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as fh:
            json.dump(candidate_meta["best_params"], fh, indent=2)
        if args.verbose:
            print(f"Wrote best params to '{args.out_json}'.")

    # ``--bounds-json``: write the narrowed recommendation when one was produced.
    rec_meta = gate.get("recommendation") or {}
    if args.bounds_json and rec_meta:
        import json

        args.bounds_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "search_space": rec_meta.get("search_space", {}),
            "best_value": rec_meta.get("best_value"),
            "n_near_optimal": rec_meta.get("n_near_optimal"),
            "margin": rec_meta.get("margin"),
        }
        with args.bounds_json.open("w") as fh:
            json.dump(payload, fh, indent=2)
        if args.verbose:
            print(f"Wrote recommended bounds to '{args.bounds_json}'.")

    # ``--apply``: write the *winning* fit (candidate on pass, baseline on
    # fallback). The safety gate guarantees this is never worse than the
    # default-bounds baseline.
    if args.apply:
        from linum_basic.fit import save_corrected

        save_corrected(mosaic, at.fit, args.apply, input_path=args.input, overwrite=True)
        if args.verbose:
            print(f"Saved corrected mosaic to '{args.apply}' (applied={at.applied}).")

    return 0


def _run_tune(args: argparse.Namespace) -> int:
    if (rc := _validate_device(args.device)) is not None:
        return rc

    from linum_basic.mosaic import MosaicGrid

    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)

    # ``--auto-apply`` runs the full D023 pipeline via auto_tune() and composes
    # with --out-json / --bounds-json / --apply / --verbose. It short-circuits
    # the manual tune()+recommend_bounds() path below to avoid double-tuning.
    if getattr(args, "auto_apply", False):
        return _run_auto_apply(args, mosaic)

    from linum_basic.tuning import recommend_bounds, tune

    run_full_fit = args.apply is not None
    result = tune(
        mosaic,
        n_trials=args.n_trials,
        z_subsample=args.z_subsample,
        seed=args.seed,
        n_workers=args.n_jobs,
        backend=args.backend,
        device=args.device,
        max_tiles=args.max_tiles if args.max_tiles > 0 else None,
        n_extra_rows=args.n_extra_rows,
        storage=args.storage,
        study_name=args.study_name,
        run_full_fit=run_full_fit,
        working_size=args.working_size,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Best value (seam_l1): {result.best_value:.6f}")
        print("Best params:")
        for k, v in result.best_params.items():
            print(f"  {k}: {v}")

    if args.out_json:
        import json

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as fh:
            json.dump(result.best_params, fh, indent=2)
        if args.verbose:
            print(f"Wrote best params to '{args.out_json}'.")

    if args.bounds_json:
        import json

        try:
            rec = recommend_bounds(result, margin=args.bounds_margin)
        except ValueError as exc:
            print(f"error: could not recommend bounds: {exc}", file=sys.stderr)
            return 1
        args.bounds_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "search_space": rec.search_space,
            "best_value": rec.best_value,
            "n_near_optimal": rec.n_near_optimal,
            "margin": rec.margin,
        }
        with args.bounds_json.open("w") as fh:
            json.dump(payload, fh, indent=2)
        if args.verbose:
            print(f"Recommended bounds (margin={rec.margin:.2f}, n_near_optimal={rec.n_near_optimal}):")
            print(json.dumps(payload["search_space"], indent=2))
            print(f"Wrote recommended bounds to '{args.bounds_json}'.")

    if args.apply and result.best_fit is not None:
        from linum_basic.fit import save_corrected

        save_corrected(mosaic, result.best_fit, args.apply, input_path=args.input, overwrite=True)
        if args.verbose:
            print(f"Saved corrected mosaic to '{args.apply}'.")

    return 0


# ---------------------------------------------------------------------------
# Sub-command: preview
# ---------------------------------------------------------------------------


def _add_preview_subcommand(subs: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subs.add_parser(
        "preview",
        help="Render an average-intensity-projection PNG preview of an OME-Zarr volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("I/O")
    io.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr volume.")
    io.add_argument("--output", metavar="PNG", required=True, type=Path, help="Path to write the preview PNG.")

    proj = p.add_argument_group("Projection")
    proj.add_argument("--axis", metavar="N", type=int, default=0, help="Axis to average over (0 = depth/z).")
    proj.add_argument(
        "--percentile", metavar="P", type=float, default=99.5, help="Upper display percentile for contrast (0-100)."
    )
    proj.add_argument("--cmap", metavar="NAME", default="viridis", help="Matplotlib colormap name.")
    proj.add_argument("--title", metavar="TEXT", default=None, help="Optional figure title.")
    proj.add_argument("--dpi", metavar="N", type=int, default=200, help="Output resolution in dots per inch.")

    p.add_argument("--verbose", action="store_true", default=False, help="Print progress information.")


def _run_preview(args: argparse.Namespace) -> int:
    from linum_basic import viz
    from linum_basic.io.zarr import load_ome_zarr

    volume, axes, scale = load_ome_zarr(args.input)

    pixel_size_mm: float | None = None
    in_plane = [s for i, s in enumerate(scale) if i != args.axis]
    if in_plane:
        pixel_size_mm = float(in_plane[-1])

    # Materialise the volume: aip_preview computes an average-intensity
    # projection over the whole array, so it needs a dense ndarray regardless
    # of whether load_ome_zarr returned one eagerly or a lazy zarr.Array handle.
    import numpy as np

    fig = viz.aip_preview(
        np.asarray(volume),
        axis=args.axis,
        pixel_size_mm=pixel_size_mm,
        cmap=args.cmap,
        title=args.title,
        percentile=args.percentile,
    )
    viz.save_figure(fig, args.output, dpi=args.dpi)

    if args.verbose:
        print(f"Saved preview to '{args.output}' (axes={axes}, scale={scale}).")

    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``basic`` command.

    Dispatches to one of four sub-commands: ``correct``, ``fit``, ``tune``,
    or ``preview``.

    Parameters
    ----------
    argv : list of str or None
        Argument list.  ``None`` reads from ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).

    Examples
    --------
    Run from the shell::

        basic correct \\
            --input /data/tiles \\
            --output /data/corrected \\
            --estimate-darkfield \\
            --backend auto \\
            --verbose
    """
    parser = argparse.ArgumentParser(
        prog="basic",
        description="Linum BaSiC shading correction toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subs = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    subs.required = True

    _add_correct_subcommand(subs)
    _add_fit_subcommand(subs)
    _add_tune_subcommand(subs)
    _add_preview_subcommand(subs)

    args = parser.parse_args(argv)

    dispatch = {
        "correct": _run_correct,
        "fit": _run_fit,
        "tune": _run_tune,
        "preview": _run_preview,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    sys.exit(main())
