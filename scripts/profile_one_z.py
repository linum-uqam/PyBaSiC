#!/usr/bin/env python3
"""Profile one z-plane fit: numpy vs torch eager vs torch compile."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--z", type=int, default=0)
    p.add_argument("--tile-fov-mm", type=float, default=0.875)
    args = p.parse_args()

    from linum_basic.io.zarr import load_ome_zarr
    from linum_basic.mosaic import MosaicGrid

    array, _axes, scale = load_ome_zarr(args.input)
    pixel_size_mm = float(scale[1])
    tile_px = round(args.tile_fov_mm / pixel_size_mm)
    tile_shape = (tile_px, tile_px)
    th, tw = tile_shape
    h_crop = (array.shape[1] // th) * th
    w_crop = (array.shape[2] // tw) * tw
    array = np.asarray(array[:, :h_crop, :w_crop], dtype=np.float32)
    mosaic = MosaicGrid(array=array, tile_shape=tile_shape)
    tiles = mosaic.iter_tiles(args.z)
    print(f"z={args.z} tiles={tiles.shape[0]} tile={tile_shape} ws=128")

    base_kw = {
        "estimate_darkfield": True,
        "working_size": 128,
        "max_reweighting_iterations": 15,
        "alm_max_iter": 100,
        "verbose": False,
    }

    def run(label: str, backend: str, device: str | None = None, compile_mode: str = "default") -> None:
        from linum_basic.core import BaSiC

        if compile_mode in {"off", "disable", "disabled"}:
            os.environ["LINUM_BASIC_ALM_COMPILE_MODE"] = "off"
        else:
            os.environ["LINUM_BASIC_ALM_COMPILE_MODE"] = compile_mode

        init = {"estimate_darkfield": True, "verbose": False, "backend": backend}
        if device:
            init["device"] = device
        t0 = time.perf_counter()
        model = BaSiC(tiles, **init)
        for k, v in base_kw.items():
            if k not in init:
                setattr(model, k, v)
        model.prepare()
        model.run()
        dt = time.perf_counter() - t0
        print(
            f"{label:20s} compile={os.environ['LINUM_BASIC_ALM_COMPILE_MODE']:8s} {dt:6.2f}s "
            f"reweight={model.reweighting_iteration} alm_last={model.last_alm_iterations} "
            f"l_s={model.l_s:.4f}"
        )

    run("numpy", backend="numpy")
    run("torch_eager", backend="torch", device="cuda:0", compile_mode="off")
    run("torch_compile", backend="torch", device="cuda:0", compile_mode="default")
    run("torch_compile_2", backend="torch", device="cuda:0", compile_mode="default")


if __name__ == "__main__":
    main()
