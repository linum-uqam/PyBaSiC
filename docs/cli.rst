.. _cli:

CLI Reference
=============

``basic_shading_correction``
----------------------------

The primary command-line interface for single-stack shading correction.

.. argparse::
   :module: linum_basic.cli
   :func: _build_arg_parser
   :prog: basic_shading_correction

Examples
^^^^^^^^

Correct a directory of TIFF tiles::

    basic_shading_correction \
        --input /path/to/tiles \
        --output /path/to/corrected

Enable dark-field estimation and GPU acceleration::

    basic_shading_correction \
        --input /path/to/tiles \
        --output /path/to/corrected \
        --estimate-darkfield \
        --backend auto \
        --verbose

``basic_fit``
-------------

Fit BaSiC flat-fields and dark-fields on an OME-Zarr mosaic grid and write
the corrected mosaic to a new store.

.. argparse::
   :module: linum_basic.cli
   :func: _build_fit_parser
   :prog: basic_fit

Examples
^^^^^^^^

Fit per-z flat-fields with dark-field estimation::

    basic_fit \
        --input mosaic.ome.zarr \
        --output corrected.ome.zarr \
        --estimate-darkfield \
        --field-mode per-z \
        --verbose

Save the raw estimated fields alongside the corrected mosaic::

    basic_fit \
        --input mosaic.ome.zarr \
        --output corrected.ome.zarr \
        --save-fields ./fields \
        --verbose

``basic_tune``
--------------

Search for optimal BaSiC regularisation weights using Optuna, minimising the
seam-consistency L1 metric over a subsample of z-levels.

.. argparse::
   :module: linum_basic.cli
   :func: _build_tune_parser
   :prog: basic_tune

Examples
^^^^^^^^

Run 50 tuning trials and write the best parameters to JSON::

    basic_tune \
        --input mosaic.ome.zarr \
        --n-trials 50 \
        --out-json best_params.json \
        --verbose

Tune and immediately apply the best parameters::

    basic_tune \
        --input mosaic.ome.zarr \
        --n-trials 100 \
        --apply corrected.ome.zarr \
        --verbose

Distributed tuning with persistent SQLite storage::

    # Launch four parallel workers (each running 25 trials)
    basic_tune --input mosaic.ome.zarr \
               --n-trials 25 \
               --storage sqlite:///tune.db \
               --study-name my-mosaic-tuning \
               --n-jobs 1 &  # repeat four times

``basic_preview``
-----------------

Render a 2-D average-intensity projection (AIP) of an OME-Zarr volume as a PNG
for a quick visual check of processed data.

.. argparse::
   :module: linum_basic.cli
   :func: _build_preview_parser
   :prog: basic_preview

Examples
^^^^^^^^

Project a corrected volume along its depth axis::

    basic_preview \
        --input corrected.ome.zarr \
        --output preview.png

Project along a different axis with a custom contrast percentile::

    basic_preview \
        --input corrected.ome.zarr \
        --output preview.png \
        --axis 1 \
        --percentile 99.9

