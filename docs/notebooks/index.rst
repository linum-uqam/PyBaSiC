Example Notebooks
=================

The notebooks below demonstrate end-to-end usage of **linum-basic**.
They are executed automatically during the documentation build so every
output you see is produced by the current version of the package.

To run them locally, install the ``notebooks`` extra:

.. code-block:: bash

   uv sync --extra notebooks
   uv run jupyter notebook docs/notebooks/01_quickstart.ipynb

.. toctree::
   :maxdepth: 1

   01_quickstart
   02_darkfield_and_tuning
   03_mosaic_and_seams
   04_advanced
   basic_usage
