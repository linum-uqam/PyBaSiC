r"""Command-line interface for BaSiC shading correction.

Entry point registered as ``basic_shading_correction`` in *pyproject.toml*,
replacing the legacy ``scripts/basic_shading_correction.py`` script.

Usage
-----
::

    basic_shading_correction --input /path/to/tiles --output /path/to/output
    basic_shading_correction --input /path/to/tiles --output /path/to/output \\
        --estimate-darkfield --backend torch --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the BaSiC CLI.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser with all BaSiC options.

    Notes
    -----
    The parser includes the following groups:

    * **I/O** — ``--input``, ``--output``, ``--extension``.
    * **Algorithm** — ``--estimate-darkfield``.
    * **Compute** — ``--backend``, ``--device``.
    * **Misc** — ``--verbose``.
    """
    parser = argparse.ArgumentParser(
        prog="basic_shading_correction",
        description="Estimate and apply BaSiC flat-field/dark-field shading correction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument(
        "--input",
        metavar="DIR",
        required=True,
        type=Path,
        help="Directory containing the input image stack.",
    )
    io_group.add_argument(
        "--output",
        metavar="DIR",
        required=True,
        type=Path,
        help="Directory to write corrected images.",
    )
    io_group.add_argument(
        "--extension",
        metavar="EXT",
        default=".tif",
        help="File extension filter (e.g. '.tif', '.png').",
    )

    # --- Algorithm ---
    alg_group = parser.add_argument_group("Algorithm")
    alg_group.add_argument(
        "--estimate-darkfield",
        action="store_true",
        default=False,
        help="Estimate the dark-field in addition to the flat-field.",
    )

    # --- Compute ---
    compute_group = parser.add_argument_group("Compute")
    compute_group.add_argument(
        "--backend",
        choices=["numpy", "torch", "auto"],
        default="numpy",
        help=("Array backend for the ALM optimisation loop.  'auto' selects Torch with CUDA/MPS when available."),
    )
    compute_group.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help=("PyTorch device string (e.g. 'cuda:0', 'mps', 'cpu').  Ignored when --backend=numpy."),
    )

    # --- Misc ---
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress bars and iteration statistics.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    r"""
    Entry point for the ``basic_shading_correction`` command.

    Parses command-line arguments, runs the BaSiC estimator on the input
    image stack, then saves the shading-corrected images to the output
    directory.

    Parameters
    ----------
    argv : list of str or None
        Argument list.  ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 on success, non-zero on error).

    Examples
    --------
    Run from the shell::

        basic_shading_correction \\
            --input /data/tiles \\
            --output /data/corrected \\
            --estimate-darkfield \\
            --backend auto \\
            --verbose
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        parser.error(f"--input '{input_dir}' is not an existing directory.")

    from linum_basic.core import BaSiC

    model = BaSiC(
        input_dir,
        estimate_darkfield=args.estimate_darkfield,
        extension=args.extension,
        verbose=args.verbose,
        backend=args.backend,
        device=args.device,
    )
    model.prepare()
    model.run()
    model.write_images(output_dir)

    if args.verbose:
        print(f"Saved corrected images to '{output_dir}'.")

    return 0


# ---------------------------------------------------------------------------
# basic_fit entry point
# ---------------------------------------------------------------------------


def _build_fit_parser() -> argparse.ArgumentParser:
    """Argument parser for ``basic_fit``."""
    parser = argparse.ArgumentParser(
        prog="basic_fit",
        description="Fit BaSiC flat/dark-fields on an OME-Zarr mosaic grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr mosaic.")
    io_group.add_argument(
        "--output", metavar="ZARR", required=True, type=Path, help="Path to write the corrected .ome.zarr mosaic."
    )
    io_group.add_argument(
        "--save-fields", metavar="DIR", default=None, type=Path, help="Directory to save flat/dark-field arrays as .npy files."
    )

    alg_group = parser.add_argument_group("Algorithm")
    alg_group.add_argument("--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction (0-1).")
    alg_group.add_argument(
        "--estimate-darkfield",
        action="store_true",
        default=False,
        help="Estimate the dark-field in addition to the flat-field.",
    )
    alg_group.add_argument(
        "--field-mode",
        choices=["per-z", "global"],
        default="per-z",
        help=("'per-z': fit one field per z-level (recommended for OCT). 'global': average all per-z fields."),
    )
    alg_group.add_argument(
        "--z-indices", metavar="Z", nargs="+", type=int, default=None, help="Z-levels to fit (default: all)."
    )

    compute_group = parser.add_argument_group("Compute")
    compute_group.add_argument(
        "--backend", choices=["numpy", "torch", "auto"], default="numpy", help="Array backend for the ALM loop."
    )
    compute_group.add_argument(
        "--device", metavar="DEVICE", default=None, help="PyTorch device string (ignored for --backend=numpy)."
    )
    compute_group.add_argument(
        "--n-jobs",
        metavar="N",
        type=int,
        default=None,
        help="Worker processes for parallel z-level fitting (default: CPU count - 2; forced to 1 on GPU).",
    )

    parser.add_argument("--verbose", action="store_true", default=False, help="Show progress bars.")
    return parser


def fit_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``basic_fit`` command.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments.  ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _build_fit_parser()
    args = parser.parse_args(argv)

    from linum_basic.fit import fit_mosaic, save_corrected
    from linum_basic.mosaic import MosaicGrid

    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)

    basic_kwargs: dict = {"estimate_darkfield": args.estimate_darkfield, "backend": args.backend, "device": args.device}
    fit = fit_mosaic(
        mosaic,
        z_indices=args.z_indices,
        field_mode=args.field_mode,
        basic_kwargs=basic_kwargs,
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
        print(f"Saved corrected mosaic to '{args.output}'.")

    return 0


# ---------------------------------------------------------------------------
# basic_tune entry point
# ---------------------------------------------------------------------------


def _build_tune_parser() -> argparse.ArgumentParser:
    """Argument parser for ``basic_tune``."""
    parser = argparse.ArgumentParser(
        prog="basic_tune",
        description="Tune BaSiC hyperparameters on an OME-Zarr mosaic using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr mosaic.")
    io_group.add_argument(
        "--out-json", metavar="FILE", default=None, type=Path, help="Write best hyperparameters as JSON to this file."
    )
    io_group.add_argument(
        "--apply", metavar="ZARR", default=None, type=Path, help="If set, run a full-z fit with the best params and save here."
    )

    tuning_group = parser.add_argument_group("Tuning")
    tuning_group.add_argument("--n-trials", metavar="N", type=int, default=50, help="Number of Optuna trials.")
    tuning_group.add_argument("--z-subsample", metavar="N", type=int, default=4, help="Z-levels evaluated per trial.")
    tuning_group.add_argument("--storage", metavar="URL", default=None, help="Optuna storage URL, e.g. sqlite:///tune.db.")
    tuning_group.add_argument(
        "--study-name", metavar="NAME", default="basic-tune", help="Optuna study name (for persistent storage)."
    )
    tuning_group.add_argument("--seed", metavar="N", type=int, default=0, help="Random seed for reproducibility.")
    tuning_group.add_argument(
        "--n-jobs",
        metavar="N",
        type=int,
        default=None,
        help="Worker threads for z-level evaluation within each trial (default: CPU count - 2; 1 enables pruning).",
    )
    tuning_group.add_argument(
        "--max-tiles",
        metavar="N",
        type=int,
        default=64,
        help="Tiles (evenly spaced) used for the seam metric per trial. Use 0 for all tiles.",
    )
    tuning_group.add_argument(
        "--n-extra-rows",
        metavar="N",
        type=int,
        default=0,
        help="Leading rows per tile to drop before fitting (galvo fly-back artefact).",
    )
    tuning_group.add_argument(
        "--overlap", metavar="FRAC", type=float, default=0.2, help="Physical tile-overlap fraction (0-1)."
    )

    parser.add_argument("--verbose", action="store_true", default=False, help="Enable Optuna logging and progress bars.")
    return parser


def tune_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``basic_tune`` command.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments.  ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _build_tune_parser()
    args = parser.parse_args(argv)

    from linum_basic.mosaic import MosaicGrid
    from linum_basic.tuning import tune

    mosaic = MosaicGrid.from_ome_zarr(str(args.input), overlap_fraction=args.overlap)

    run_full_fit = args.apply is not None
    result = tune(
        mosaic,
        n_trials=args.n_trials,
        z_subsample=args.z_subsample,
        seed=args.seed,
        n_workers=args.n_jobs,
        max_tiles=args.max_tiles if args.max_tiles > 0 else None,
        n_extra_rows=args.n_extra_rows,
        storage=args.storage,
        study_name=args.study_name,
        run_full_fit=run_full_fit,
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

    if args.apply and result.best_fit is not None:
        from linum_basic.fit import save_corrected

        save_corrected(mosaic, result.best_fit, args.apply, input_path=args.input, overwrite=True)
        if args.verbose:
            print(f"Saved corrected mosaic to '{args.apply}'.")

    return 0


# ---------------------------------------------------------------------------
# basic_preview entry point
# ---------------------------------------------------------------------------


def _build_preview_parser() -> argparse.ArgumentParser:
    """Argument parser for ``basic_preview``."""
    parser = argparse.ArgumentParser(
        prog="basic_preview",
        description="Render an average-intensity-projection PNG preview of an OME-Zarr volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("--input", metavar="ZARR", required=True, type=Path, help="Path to the input .ome.zarr volume.")
    io_group.add_argument("--output", metavar="PNG", required=True, type=Path, help="Path to write the preview PNG.")

    proj_group = parser.add_argument_group("Projection")
    proj_group.add_argument("--axis", metavar="N", type=int, default=0, help="Axis to average over (0 = depth/z).")
    proj_group.add_argument(
        "--percentile", metavar="P", type=float, default=99.5, help="Upper display percentile for contrast (0-100)."
    )
    proj_group.add_argument("--cmap", metavar="NAME", default="viridis", help="Matplotlib colormap name.")
    proj_group.add_argument("--title", metavar="TEXT", default=None, help="Optional figure title.")
    proj_group.add_argument("--dpi", metavar="N", type=int, default=200, help="Output resolution in dots per inch.")

    parser.add_argument("--verbose", action="store_true", default=False, help="Print progress information.")
    return parser


def preview_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``basic_preview`` command.

    Renders a 2-D average-intensity projection of an OME-Zarr volume as a PNG,
    suitable for a quick visual check of processed data.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments.  ``None`` reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    parser = _build_preview_parser()
    args = parser.parse_args(argv)

    from linum_basic import viz
    from linum_basic.io.zarr import load_ome_zarr

    volume, axes, scale = load_ome_zarr(args.input)

    # In-plane pixel size (mm) from the non-projected spatial axes.
    pixel_size_mm: float | None = None
    in_plane = [s for i, s in enumerate(scale) if i != args.axis]
    if in_plane:
        pixel_size_mm = float(in_plane[-1])

    fig = viz.aip_preview(
        volume,
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


if __name__ == "__main__":
    sys.exit(main())
