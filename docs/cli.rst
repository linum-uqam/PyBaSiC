.. _cli:

CLI Reference
=============

All sub-commands are exposed through a single ``basic`` executable:

.. code-block:: text

    basic <COMMAND> [options]

Run ``basic --help`` or ``basic <COMMAND> --help`` for a full option listing.

``basic correct``
-----------------

Estimate and apply BaSiC flat-field / dark-field shading correction to a
directory of image files.

.. code-block:: text

    basic correct --input DIR --output DIR [options]

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--input DIR``
     - *required*
     - Directory containing the input image stack.
   * - ``--output DIR``
     - *required*
     - Directory to write corrected images.
   * - ``--extension EXT``
     - ``.tif``
     - File extension filter.
   * - ``--estimate-darkfield``
     - off
     - Estimate the dark-field in addition to the flat-field.
   * - ``--backend {numpy,torch,auto}``
     - ``numpy``
     - ALM compute backend.
   * - ``--device DEVICE``
     - ``None``
     - PyTorch device string (e.g. ``cuda:0``).
   * - ``--verbose``
     - off
     - Print progress bars.

Examples
^^^^^^^^

Correct a directory of TIFF tiles::

    basic correct \
        --input /path/to/tiles \
        --output /path/to/corrected

Enable dark-field estimation and GPU acceleration::

    basic correct \
        --input /path/to/tiles \
        --output /path/to/corrected \
        --estimate-darkfield \
        --backend auto \
        --verbose

``basic fit``
-------------

Fit BaSiC flat-fields and dark-fields on an OME-Zarr mosaic grid and write the
corrected mosaic to a new store.

.. code-block:: text

    basic fit --input ZARR --output ZARR [options]

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--input ZARR``
     - *required*
     - Input ``.ome.zarr`` mosaic path.
   * - ``--output ZARR``
     - *required*
     - Output corrected ``.ome.zarr`` path.
   * - ``--save-fields DIR``
     - ``None``
     - Save flat/dark-field arrays as ``.npy`` files here.
   * - ``--overlap FRAC``
     - ``0.2``
     - Physical tile-overlap fraction (0–1).
   * - ``--estimate-darkfield``
     - off
     - Estimate the dark-field.
   * - ``--field-mode {per-z,global}``
     - ``per-z``
     - Fit one field per z-level, or average into a single global field.
   * - ``--z-indices Z [Z ...]``
     - all
     - Z-levels to fit.
   * - ``--backend {numpy,torch,auto}``
     - ``numpy``
     - ALM compute backend.
   * - ``--device DEVICE``
     - ``None``
     - PyTorch device string.
   * - ``--n-jobs N``
     - auto
     - Worker processes for parallel z-level fitting.
   * - ``--verbose``
     - off
     - Show progress bars.

Examples
^^^^^^^^

Fit per-z flat-fields with dark-field estimation::

    basic fit \
        --input mosaic.ome.zarr \
        --output corrected.ome.zarr \
        --estimate-darkfield \
        --field-mode per-z \
        --verbose

Save the raw estimated fields alongside the corrected mosaic::

    basic fit \
        --input mosaic.ome.zarr \
        --output corrected.ome.zarr \
        --save-fields ./fields \
        --verbose

``basic tune``
--------------

Search for optimal BaSiC regularisation weights using Optuna, minimising the
seam-consistency L1 metric over a subsample of z-levels.

.. code-block:: text

    basic tune --input ZARR [options]

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--input ZARR``
     - *required*
     - Input ``.ome.zarr`` mosaic path.
   * - ``--out-json FILE``
     - ``None``
     - Write best hyperparameters as JSON.
   * - ``--apply ZARR``
     - ``None``
     - Run a full-z fit with the best params and save here.
   * - ``--bounds-json FILE``
     - ``None``
     - Write the recommended narrowed search-space bounds (``recommend_bounds``) as JSON.
   * - ``--auto-apply``
     - off
     - Run the auto-apply safety gate: fit one default-bounds baseline and one narrowed-bounds candidate, gate the candidate on full-volume ``seam_l1`` + ``seam_curvature`` deltas (fixed 0.0 non-regression margin), and write the winning fit to ``--apply`` (or report the fallback). Independent of ``--bounds-json`` / ``--bounds-margin`` / ``--out-json`` / ``--apply``.
   * - ``--n-trials N``
     - ``50``
     - Number of Optuna trials.
   * - ``--z-subsample N``
     - ``4``
     - Z-levels evaluated per trial.
   * - ``--storage URL``
     - ``None``
     - Optuna storage URL (e.g. ``sqlite:///tune.db``).
   * - ``--study-name NAME``
     - ``basic-tune``
     - Optuna study name for persistent storage.
   * - ``--seed N``
     - ``0``
     - Random seed for reproducibility.
   * - ``--n-jobs N``
     - auto
     - Worker threads per trial.
   * - ``--max-tiles N``
     - ``64``
     - Tiles used for the seam metric.  ``0`` = all tiles.
   * - ``--n-extra-rows N``
     - ``0``
     - Leading rows to drop per tile (galvo fly-back artefact).
   * - ``--overlap FRAC``
     - ``0.2``
     - Physical tile-overlap fraction.
   * - ``--bounds-margin FRAC``
     - ``0.10``
     - Relative margin (0-1) selecting the near-optimal trial band used by ``--bounds-json``.
   * - ``--backend {numpy,torch}``
     - ``numpy``
     - Compute backend.
   * - ``--device DEV``
     - ``None``
     - Torch device string.
   * - ``--verbose``
     - off
     - Enable Optuna logging and progress bars.

Examples
^^^^^^^^

Run 50 tuning trials and write the best parameters to JSON::

    basic tune \
        --input mosaic.ome.zarr \
        --n-trials 50 \
        --out-json best_params.json \
        --verbose

Tune and immediately apply the best parameters::

    basic tune \
        --input mosaic.ome.zarr \
        --n-trials 100 \
        --apply corrected.ome.zarr \
        --verbose

Tune, then write recommended narrowed bounds for a follow-up ``tune`` run::

    basic tune \
        --input mosaic.ome.zarr \
        --n-trials 50 \
        --bounds-json bounds.json \
        --bounds-margin 0.10 \
        --verbose

The ``bounds.json`` payload carries a ``search_space`` dict (in the
scale-invariant ``l_s_divisor`` / ``l_d_divisor`` parametrisation) that plugs
straight back into a follow-up ``tune`` call via the library's ``tune(...)``
``search_space=`` argument.

Tune and auto-apply under the safety gate (D023)::

    basic tune \
        --input mosaic.ome.zarr \
        --n-trials 50 \
        --auto-apply \
        --apply corrected.ome.zarr \
        --verbose

With ``--auto-apply``, the command runs the full pipeline — a default-bounds
baseline fit, one Optuna study, a narrowed-bounds candidate fit, and a
non-regression gate on full-volume ``seam_l1`` + ``seam_curvature`` — then
writes the *winning* fit (the candidate on a pass, the baseline on any
fallback) to ``--apply``. ``--auto-apply`` composes with ``--out-json``
(candidate best params), ``--bounds-json`` (the narrowed recommendation), and
``--verbose`` (which prints the gate verdict, failing metrics, fallback
reason, and per-metric deltas). If the default-bounds baseline fit itself
fails, the command exits 1 with an ``AutoApplyError`` message rather than
writing an undefined correction. See :doc:`auto_apply_safety_gate` for the
full design contract.

Distributed tuning with persistent SQLite storage::

    # Launch four parallel workers (each running 25 trials)
    basic tune --input mosaic.ome.zarr \
               --n-trials 25 \
               --storage sqlite:///tune.db \
               --study-name my-mosaic-tuning \
               --n-jobs 1 &  # repeat four times

``basic preview``
-----------------

Render a 2-D average-intensity projection (AIP) of an OME-Zarr volume as a PNG
for a quick visual check of processed data.

.. code-block:: text

    basic preview --input ZARR --output PNG [options]

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Option
     - Default
     - Description
   * - ``--input ZARR``
     - *required*
     - Input ``.ome.zarr`` volume path.
   * - ``--output PNG``
     - *required*
     - Output PNG path.
   * - ``--axis N``
     - ``0``
     - Axis to average over (0 = depth/z).
   * - ``--percentile P``
     - ``99.5``
     - Upper display percentile for contrast.
   * - ``--cmap NAME``
     - ``viridis``
     - Matplotlib colormap name.
   * - ``--title TEXT``
     - ``None``
     - Optional figure title.
   * - ``--dpi N``
     - ``200``
     - Output resolution in dots per inch.
   * - ``--verbose``
     - off
     - Print progress information.

Examples
^^^^^^^^

Project a corrected volume along its depth axis::

    basic preview \
        --input corrected.ome.zarr \
        --output preview.png

Project along a different axis with a custom contrast percentile::

    basic preview \
        --input corrected.ome.zarr \
        --output preview.png \
        --axis 1 \
        --percentile 99.9

