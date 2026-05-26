.. _cli:

CLI Reference
=============

The ``basic_shading_correction`` command is the primary command-line
interface for linum-basic, registered as a script entry point.

.. argparse::
   :module: linum_basic.cli
   :func: _build_arg_parser
   :prog: basic_shading_correction

Examples
--------

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
