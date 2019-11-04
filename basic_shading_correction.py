#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import cv2
import numpy as np
from pybasic.shading_correction import BaSiC

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Shading correction algorithm based on the BaSiC method")
    parser.add_argument("directory", help="Full path a directory containing the tiles to process")
    parser.add_argument("output_directory", help="Full path to a directory where the output will be saved")
    parser.add_argument("--extension", default=".tif", help="Image tile extension (default=%(default)s)")
    parser.add_argument("--estimate_darkfield", action="store_true", help="Estimate the darkfield in addition to the flatfield.")
    parser.add_argument("--apply_correction", action="store_true", help="Apply the shading correction with the estimated or loaded profiles.")
    parser.add_argument("--use_flatfield", default=None, help="Use existing flatfield (skip estimation)")
    parser.add_argument("--use_darkfield", default=None, help="Use existing darkfield (skip estimation)")
    parser.add_argument("--epsilon", default=1e-6, type=float, help="Stability coefficient to use for the shading correction. (default=%(default)s)")
    parser.add_argument("--l_s", default=None, type=float,
                              help="Flat-field regularization parameter (set automatically if None)")
    parser.add_argument("--l_d", default=None, type=float,
                              help="Dark-field regularization parameter (set automatically if None)")
    parser.add_argument("--output_flatfield_filename", default=None, help="Optional output flatfield filename (if none, flatfield.tif will be saved in the output directory).")
    parser.add_argument("--output_darkfield_filename", default=None, help="Optional output flatfield filename (if none, darkfield.tif will be saved in the output directory).")
    args = parser.parse_args()

    # Create the BaSiC shading correction object
    optimizer = BaSiC(args.directory, estimate_darkfield=args.estimate_darkfield, extension=args.extension)

    # Set some optimizer parameters
    optimizer.l_s = args.l_s
    optimizer.l_d = args.l_d

    # Prepare the optimization
    optimizer.prepare()

    # Extract the flat and dark fields
    perform_estimation = True
    if args.use_flatfield is not None:
        img = cv2.imread(args.use_flatfield, cv2.IMREAD_ANYDEPTH)
        optimizer.set_flatfield(img)
        perform_estimation = False

    if args.use_darkfield is not None:
        img = cv2.imread(args.use_darkfield, cv2.IMREAD_ANYDEPTH)
        optimizer.set_flatfield(img)
        perform_estimation = False

    # Perform the estimation
    if perform_estimation:
        optimizer.run()

        # Save the estimated fields (only if the profiles were estimated)
        directory = Path(args.output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if args.output_flatfield_filename is not None:
            flatfield_name = Path(args.output_flatfield_filename).resolve()
            flatfield_name.parent.mkdir(parents=True, exist_ok=True)
        else:
            flatfield_name = directory / "flatfield.tif"
        if args.output_darkfield_filename is not None:
            darkfield_name = Path(args.output_darkfield_filename).resolve()
            darkfield_name.parent.mkdir(parents=True, exist_ok=True)
        else:
            darkfield_name = directory / "darkfield.tif"

        cv2.imwrite(str(flatfield_name), optimizer.flatfield_fullsize.astype(np.float32))
        cv2.imwrite(str(darkfield_name), optimizer.darkfield_fullsize.astype(np.float32))

    # Apply shading correction.
    if args.apply_correction:
        optimizer.write_images(args.output_directory, epsilon=args.epsilon)