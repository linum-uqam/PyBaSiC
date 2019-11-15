# PyBaSiC
*Python implementation of the BaSiC shading correction method.*

* **Original paper**: T. Peng et al., “A BaSiC tool for background and shading correction of optical microscopy images,” Nat. Commun., vol. 8, p. 14836, Jun. 2017. [DOI:10.1038/ncomms14836](https://doi.org/10.1038/ncomms14836)
* **Nature Supplementary Materials**: [PDF](https://static-content.springer.com/esm/art%3A10.1038%2Fncomms14836/MediaObjects/41467_2017_BFncomms14836_MOESM560_ESM.pdf)
* **Matlab implementation**: https://github.com/QSCD/BaSiC
* **Fiji plugin**: [URL](https://www.helmholtz-muenchen.de/icb/research/groups/quantitative-single-cell-dynamics/software/basic/index.html)
* **Demo data examples**: [Dropbox](https://www.dropbox.com/s/plznvzdjglrse3h/Demoexamples.zip?dl=0)

## Installation
* Dependencies: `Anaconda3`, `Python >= 3.6`
* Clone the repository
* Install the conda environment for this module
```bash
conda env create -f environment.yml
```
* Load the environment
```bash
conda activate pybasic
```
* Install this python package.
```bash
pip install .
```
## Usage
The main script to perform the retrospective shading estimation and correction is `basic_shading_correction.py`. 
Here is a description of its arguments. **Note**: You must first activate the `pybasic` environment to run this script.

```bash
>> basic_shading_correction.py --help
usage: Shading correction algorithm based on the BaSiC method
       [-h] [--extension EXTENSION] [--estimate_darkfield]
       [--apply_correction] [--use_flatfield USE_FLATFIELD]
       [--use_darkfield USE_DARKFIELD] [--epsilon EPSILON] [--l_s L_S]
       [--l_d L_D] [--output_flatfield_filename OUTPUT_FLATFIELD_FILENAME]
       [--output_darkfield_filename OUTPUT_DARKFIELD_FILENAME]
       directory output_directory

positional arguments:
  directory             Full path a directory containing the tiles to process
  output_directory      Full path to a directory where the output will be
                        saved

optional arguments:
  -h, --help            show this help message and exit
  --extension EXTENSION
                        Image tile extension (default=.tif)
  --estimate_darkfield  Estimate the darkfield in addition to the flatfield.
  --apply_correction    Apply the shading correction with the estimated or
                        loaded profiles.
  --use_flatfield USE_FLATFIELD
                        Use existing flatfield (skip estimation)
  --use_darkfield USE_DARKFIELD
                        Use existing darkfield (skip estimation)
  --epsilon EPSILON     Stability coefficient to use for the shading
                        correction. (default=1e-06)
  --l_s L_S             Flat-field regularization parameter (set automatically
                        if None)
  --l_d L_D             Dark-field regularization parameter (set automatically
                        if None)
  --output_flatfield_filename OUTPUT_FLATFIELD_FILENAME
                        Optional output flatfield filename (if none,
                        flatfield.tif will be saved in the output directory).
  --output_darkfield_filename OUTPUT_DARKFIELD_FILENAME
                        Optional output flatfield filename (if none,
                        darkfield.tif will be saved in the output directory).
  --verbose
```
