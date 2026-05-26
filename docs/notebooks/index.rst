Example Notebooks
=================

The notebooks below demonstrate end-to-end usage of **linum-basic**.
They are executed automatically during the documentation build so every
output you see is produced by the current version of the package.

To run them locally, install the ``notebooks`` extra:

.. code-block:: bash

   uv run --with linum-basic[notebooks] jupyter notebook docs/notebooks/basic_usage.ipynb

Or, if you have the repository checked out:

.. code-block:: bash

   uv sync --extra notebooks
   uv run jupyter notebook docs/notebooks/basic_usage.ipynb

.. toctree::
   :maxdepth: 1

   basic_usage
