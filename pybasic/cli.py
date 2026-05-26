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
        help=(
            "Array backend for the ALM optimisation loop.  "
            "'auto' selects Torch with CUDA/MPS when available."
        ),
    )
    compute_group.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help=(
            "PyTorch device string (e.g. 'cuda:0', 'mps', 'cpu').  Ignored when --backend=numpy."
        ),
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

    from pybasic.core import BaSiC  # noqa: PLC0415

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


if __name__ == "__main__":
    sys.exit(main())
