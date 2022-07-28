import json
import pathlib
from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd

from .boundary import WatershedCellBoundary
from .image import MerfishMosaicImage
from .transform import MerfishTransform


class MerfishExperimentRegion:
    """Entry point for one region of a MERFISH experiment."""

    def __init__(self, region_dir):
        self.region_dir = pathlib.Path(region_dir).absolute()

        # image coordinates transform
        # micron to pixel transform
        self.micron_to_pixel_transform_path = region_dir / "images/micron_to_mosaic_pixel_transform.csv"
        self.transform = MerfishTransform(self.micron_to_pixel_transform_path)

        # image manifest
        self.image_manifest = self._read_image_manifest()

        # image paths
        self._mosaic_image_zarr_paths = {
            p.name.split(".")[0].split("_")[-1]: p for p in region_dir.glob("images/*.zarr")
        }
        self._opened_images = {}

        # prepare fov table
        self.fov_table = self._prepare_fov_table()
        # columns
        # ['x', 'y', 'n_x', 'n_y',
        #  'start_x_micron', 'start_y_micron',
        #  'end_x_micron', 'end_y_micron',
        #  'start_x_pixel', 'start_y_pixel',
        #  'end_x_pixel', 'end_y_pixel']

        # watershed cell boundaries
        cell_boundary_dir = region_dir / "cell_boundaries"
        self._cell_boundary_hdf_paths = {
            int(p.name.split(".")[0].split("_")[-1]): p for p in pathlib.Path(cell_boundary_dir).glob("*.hdf5")
        }
        # {fov int id: fov hdf5 path}
        return

    def _read_image_manifest(self):
        with open(self.region_dir / "images/manifest.json") as f:
            manifest = json.load(f)
        manifest = namedtuple("ImageManifest", manifest.keys())(*manifest.values())
        return manifest

    def get_fov_coords_pixel(self, fov):
        """Get fov coordinates in pixel."""
        xmin, xmax, ymin, ymax = self.fov_table.loc[
            fov, ["start_x_pixel", "end_x_pixel", "start_y_pixel", "end_y_pixel"]
        ]
        return xmin, xmax, ymin, ymax

    def get_fov_coords_micron(self, fov):
        """Get fov coordinates in micron."""
        xmin, xmax, ymin, ymax = self.fov_table.loc[
            fov, ["start_x_micron", "end_x_micron", "start_y_micron", "end_y_micron"]
        ]
        return xmin, xmax, ymin, ymax

    def get_image(self, name, fov: Union[str, int] = None, z: Union[int, slice] = None):
        """Get image by name, and select fov (optional) and z slice (optional)."""
        if name not in self._opened_images:
            try:
                zarr_path = self._mosaic_image_zarr_paths[name]
            except KeyError:
                raise KeyError(f"Do not have {name} image, possible names are {self._mosaic_image_zarr_paths.keys()}")
            img = MerfishMosaicImage(zarr_path)
            self._opened_images[name] = img
        else:
            img = self._opened_images[name]

        if z is None:
            z = slice(None)

        if fov is not None:
            xmin, xmax, ymin, ymax = self.get_fov_coords_pixel(fov)
            values = img[z, ymin:ymax, xmin:xmax].values
        else:
            values = img[z, :, :].values
        return values

    def _prepare_fov_table(self):
        fov_tiles = pd.read_csv(self.region_dir / "../../raw/settings/positions.csv", header=None, names=["x", "y"])

        # order x, y tiles
        fov_tiles.index.name = "fov"
        x_to_tile = {v: i for i, v in enumerate(fov_tiles["x"].sort_values().unique())}
        y_to_tile = {v: i for i, v in enumerate(fov_tiles["y"].sort_values().unique())}
        fov_tiles["n_x"] = fov_tiles["x"].map(x_to_tile)
        fov_tiles["n_y"] = fov_tiles["y"].map(y_to_tile)

        # microscope setting and global coordinates
        (
            global_xmin_micron,
            global_ymin_micron,
            global_xmax_micron,
            global_ymax_micron,
        ) = self.image_manifest.bbox_microns
        global_xmin_pixel, global_ymin_pixel, global_xmax_pixel, global_ymax_pixel = (
            0,
            0,
            self.image_manifest.mosaic_width_pixels,
            self.image_manifest.mosaic_height_pixels,
        )

        x_tiles = self.image_manifest.hor_num_tiles_box
        y_tiles = self.image_manifest.vert_num_tiles_box

        tile_width_micron = (global_xmax_micron - global_xmin_micron) / x_tiles
        tile_height_micron = (global_ymax_micron - global_ymin_micron) / y_tiles
        tile_width_pixel = (global_xmax_pixel - global_xmin_pixel) / x_tiles
        tile_height_pixel = (global_ymax_pixel - global_ymin_pixel) / y_tiles

        # For each FOV, add the start and end position
        # ------------O (end_x_*, end_y_*)
        # |           |
        # |           |
        # |    FOV    |
        # |           |
        # |           |
        # O------------
        # (start_x_*, start_y_*)

        # micron
        fov_tiles["start_x_micron"] = global_xmin_micron + fov_tiles["n_x"] * tile_width_micron
        fov_tiles["start_y_micron"] = global_ymin_micron + fov_tiles["n_y"] * tile_height_micron
        fov_tiles["end_x_micron"] = fov_tiles["start_x_micron"] + tile_width_micron
        fov_tiles["end_y_micron"] = fov_tiles["start_y_micron"] + tile_height_micron

        # pixel
        fov_tiles["start_x_pixel"] = np.round(global_xmin_pixel + fov_tiles["n_x"] * tile_width_pixel).astype(int)
        fov_tiles["start_y_pixel"] = np.round(global_ymin_pixel + fov_tiles["n_y"] * tile_height_pixel).astype(int)
        fov_tiles["end_x_pixel"] = np.round(fov_tiles["start_x_pixel"] + tile_width_pixel).astype(int)
        fov_tiles["end_y_pixel"] = np.round(fov_tiles["start_y_pixel"] + tile_height_pixel).astype(int)
        return fov_tiles
