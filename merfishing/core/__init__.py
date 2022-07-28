import json
import pathlib
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..pl.plot_image import MerfishImageAxesPlotter
from .boundary import WatershedCellBoundary
from .image import MerfishMosaicImage
from .transform import MerfishTransform


class MerfishExperimentRegion:
    """Entry point for one region of a MERFISH experiment."""

    def __init__(self, region_dir):
        self.region_dir = pathlib.Path(region_dir).absolute()

        # image coordinates transform
        # micron to pixel transform
        self.micron_to_pixel_transform_path = self.region_dir / "images/micron_to_mosaic_pixel_transform.csv"
        self.transform = MerfishTransform(self.micron_to_pixel_transform_path)

        # image manifest
        self.image_manifest = self._read_image_manifest()

        # image paths
        self._mosaic_image_zarr_paths = {
            p.name.split(".")[0].split("_")[-1]: p for p in self.region_dir.glob("images/*.zarr")
        }
        self._opened_images = {}

        self._cell_metadata = None
        self._cell_to_fov_map = None

        # watershed cell segmentation
        cell_boundary_dir = self.region_dir / "cell_boundaries"
        # {fov int id: fov hdf5 path}
        self._watershed_cell_boundary_hdf_paths = {
            int(p.name.split(".")[0].split("_")[-1]): p for p in pathlib.Path(cell_boundary_dir).glob("*.hdf5")
        }

        # transcripts
        self.transcripts_path = self.region_dir / "detected_transcripts.hdf5"

        return

    def _read_image_manifest(self):
        with open(self.region_dir / "images/manifest.json") as f:
            manifest = json.load(f)
        manifest = namedtuple("ImageManifest", manifest.keys())(*manifest.values())
        return manifest

    @property
    def image_names(self):
        """Get image names."""
        return list(self._mosaic_image_zarr_paths.keys())

    @property
    def smfish_genes(self):
        """Get smfish genes."""
        return [g for g in self._mosaic_image_zarr_paths.keys() if g not in ["DAPI", "PolyT"]]

    def get_image(self, name):
        """Get image by name, and select fov (optional) and z slice (optional)."""
        # TODO FOV coordinate system is not correct
        if name not in self._opened_images:
            try:
                zarr_path = self._mosaic_image_zarr_paths[name]
            except KeyError:
                raise KeyError(f"Do not have {name} image, possible names are {self._mosaic_image_zarr_paths.keys()}")
            img = MerfishMosaicImage(zarr_path)
            self._opened_images[name] = img
        else:
            img = self._opened_images[name]
        return img

    def get_image_fov(self, name, fov, z=None, load=True, projection=None, padding=300):
        """
        Get image data of FOV.

        Parameters
        ----------
        name :
            image name, use self.image_names to get all names
        fov :
            fov id
        z :
            select z slice
        load :
            whether to load image data
        projection :
            projection type along z axis
        padding :
            padding in pixel

        Returns
        -------
        image : np.ndarray or xr.DataArray
        """
        img = self.get_image(name)
        xmin, ymin, xmax, ymax = self.get_fov_pixel_extent_from_transcripts(fov, padding=padding)
        xslice = slice(xmin, xmax)
        yslice = slice(ymin, ymax)
        if z is None:
            z = slice(None)
        return img.get_image(z, yslice, xslice, load=load, projection=projection)

    def get_transcripts(self, fov):
        """Get transcripts detected in the FOV."""
        return pd.read_hdf(self.transcripts_path, key=str(fov))

    def get_fov_micron_extent_from_transcripts(self, fov):
        """Get fov extent in micron coords from transcripts detected in the FOV."""
        transcripts = self.get_transcripts(fov)
        xmin, ymin = transcripts[["global_x", "global_y"]].min()
        xmax, ymax = transcripts[["global_x", "global_y"]].max()
        return xmin, ymin, xmax, ymax

    def get_fov_pixel_extent_from_transcripts(self, fov, padding=0):
        """Get fov extent in pixel coords from transcripts detected in the FOV."""
        xmin, ymin, xmax, ymax = self.get_fov_micron_extent_from_transcripts(fov)
        extent = np.array([[xmin, ymin], [xmax, ymax]])
        pixel_extent = self.transform.micron_to_pixel_transform(extent)

        x_max_pixel = self.image_manifest.mosaic_width_pixels
        y_max_pixel = self.image_manifest.mosaic_height_pixels
        xmin = max(0, int(pixel_extent[0, 0] - padding))
        ymin = max(0, int(pixel_extent[0, 1] - padding))
        xmax = min(x_max_pixel, int(pixel_extent[1, 0] + padding))
        ymax = min(y_max_pixel, int(pixel_extent[1, 1] + padding))
        return xmin, ymin, xmax, ymax

    def _call_spots_fov(self, image_name, fov, **spot_kwargs):
        from ..tl.smfish import call_spot

        image = self.get_image(image_name)

        spot, *_ = call_spot(image, **spot_kwargs)
        return

    def _load_cell_boundaries(self):
        path = self.region_dir / "cell_metadata.csv.gz"
        if pathlib.Path(path).exists():
            df = pd.read_csv(path, index_col=0)
        else:
            path = self.region_dir / "cell_metadata.csv"
            if pathlib.Path(path).exists():
                df = pd.read_csv(path, index_col=0)
            else:
                raise FileNotFoundError(f"{path} not found")
        self._cell_metadata = df
        self._cell_to_fov_map = df["fov"].to_dict()

    def get_cell_metadata(self, fov=None, cell_segmentation="watershed"):
        """Get cell metadata."""
        # TODO add cell pose and use it as default
        if cell_segmentation == "watershed":
            if self._cell_metadata is None:
                self._load_cell_boundaries()
            df = self._cell_metadata
        else:
            raise NotImplementedError(f"{cell_segmentation} not implemented")

        if fov is not None:
            df = df.loc[df["fov"] == fov]
        return df

    def get_cell_boundaries(self, fov=None, cells=None):
        """Get cell boundaries."""
        if cells is None:
            if fov is None:
                raise ValueError("fov or cell must be specified")
            df = self.get_cell_metadata(fov)
            cells = df.index.to_list()

        boundaries = {}
        for cell in cells:
            hdf_path = self._watershed_cell_boundary_hdf_paths[fov]
            boundaries[cell] = WatershedCellBoundary(hdf_path=hdf_path, cell_id=cell)
        return boundaries

    def plot_fov(self, fov, genes=None, padding=150, ax_width=5):
        """Plot fov DAPI and PolyT image with transcript spots overlay."""
        cell_meta = self.get_cell_metadata(fov=fov)
        boundaries = self.get_cell_boundaries(fov=fov)

        transcripts = self.get_transcripts(fov)
        if genes is None:
            genes = []

        xmin, ymin, xmax, ymax = self.get_fov_pixel_extent_from_transcripts(fov, padding=padding)
        fov_images = {
            name: self.get_image_fov(name=name, fov=fov, projection="max", padding=padding)
            for name in ["DAPI", "PolyT"]
        }

        fig, axes = plt.subplots(figsize=(ax_width * 2, ax_width), ncols=2, nrows=1, dpi=300)

        offset = (xmin, ymin)

        ax = axes[0]
        plotter = MerfishImageAxesPlotter(
            ax=ax,
            image=fov_images["DAPI"],
            boundaries=boundaries,
            cells=cell_meta[["center_x", "center_y"]],
            transform=self.transform,
            offset=offset,
        )
        plotter.plot_image(cmap="Blues")
        plotter.plot_boundaries()
        plotter.plot_cell_centers(s=10)

        for gene in genes:
            gene_data = transcripts.loc[transcripts["gene"] == gene, ["global_x", "global_y"]]
            plotter.plot_scatters(gene_data, label=gene, s=0.5, linewidth=0)

        ax = axes[1]
        plotter = MerfishImageAxesPlotter(
            ax=ax,
            image=fov_images["PolyT"],
            boundaries=boundaries,
            cells=cell_meta[["center_x", "center_y"]],
            transform=self.transform,
            offset=offset,
        )
        plotter.plot_image(cmap="Reds")
        plotter.plot_boundaries()
        plotter.plot_cell_centers(s=10)

        for gene in genes:
            gene_data = transcripts.loc[transcripts["gene"] == gene, ["global_x", "global_y"]]
            plotter.plot_scatters(gene_data, label=gene, s=0.5, linewidth=0)
        return
