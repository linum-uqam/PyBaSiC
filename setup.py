# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

def main():
    setup(
        name             = 'PyBaSiC',
        version          = '0.0.1',
        packages         = find_packages(),
        author           = 'Joel Lefebvre',
        author_email     = 'joel.lefebvre@eng.ox.ac.uk',
        url              = "https://github.com/joe-from-mtl/PyBaSiC.git",
        description      = 'Python Implementation of the BaSiC shading correction algorithm',
        install_requires = ["numpy", "scipy", "tqdm"],
        scripts          = ["basic_shading_correction.py"],
        python_requires='>=3.6',
    )

if __name__ == "__main__":
    main()