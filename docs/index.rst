PyBaSiC
=======

**PyBaSiC** is a Python implementation of the BaSiC (Background and
Shading Correction) algorithm for optical microscopy images.  It
corrects spatially non-uniform illumination (flatfield) and background
offsets (darkfield) from fluorescence, bright-field, and other modalities.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting Started
      :link: getting_started
      :link-type: doc

      Installation, minimal working example, and CLI quick-reference.

   .. grid-item-card:: Algorithm
      :link: algorithm
      :link-type: doc

      Mathematical derivation of the BaSiC model, the ALM solver, and
      the reweighting scheme with flow diagrams.

   .. grid-item-card:: Parameter Tuning
      :link: parameters
      :link-type: doc

      Detailed guide for every tuning knob — physical meaning, default
      heuristics, and troubleshooting recipes.

   .. grid-item-card:: Library API
      :link: api/index
      :link-type: doc

      Auto-generated reference for the ``pybasic`` Python package.

   .. grid-item-card:: GPU Acceleration
      :link: gpu
      :link-type: doc

      Run BaSiC on CUDA/MPS hardware using the PyTorch backend.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Dev environment, pre-commit, make targets, and docstring conventions.


.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   algorithm
   parameters
   gpu
   validation
   contributing
   reference
   api/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
