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
        if name not in self._opened_images:
            try:
                zarr_path = self._mosaic_image_zarr_paths[name]
            except KeyError:
                raise KeyError(
                    f"Do not have {name} image, " f"possible names are {self._mosaic_image_zarr_paths.keys()}"
                )
            img = MerfishMosaicImage(zarr_path)
            self._opened_images[name] = img
        else:
            img = self._opened_images[name]
        return img

    def get_image_fov(self, name, fov, z=None, load=True, projection=None, padding=300, contrast=True):
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
        contrast :
            If True, adjust contrast. Only valid if load is True.

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
        return img.get_image(z, yslice, xslice, load=load, projection=projection, contrast=contrast)

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

    def get_rgb_image(self, r_name=None, g_name=None, b_name=None, as_float=False, **kwargs):
        """
        Get RGB image from up to three different mosaic images.

        Parameters
        ----------
        r_name :
            name of red channel image
        g_name :
            name of green channel image
        b_name :
            name of blue channel image
        as_float :
            whether to return image value as float, this is necessary for matplotlib imshow
        kwargs :
            keyword arguments for MerfishExperimentRegion.get_image_fov

        Returns
        -------
        rgb_image : np.ndarray
        """
        rgb = [None, None, None]
        shape = None
        dtype = None
        if r_name is not None:
            rgb[0] = self.get_image_fov(r_name, **kwargs)
            shape = rgb[0].shape
            dtype = rgb[0].dtype
        if g_name is not None:
            rgb[1] = self.get_image_fov(g_name, **kwargs)
            shape = rgb[1].shape
            dtype = rgb[1].dtype
        if b_name is not None:
            rgb[2] = self.get_image_fov(b_name, **kwargs)
            shape = rgb[2].shape
            dtype = rgb[2].dtype
        if shape is None:
            raise ValueError("At least one of r_name, g_name, b_name must be specified")
        rgb = np.array([data if data is not None else np.zeros(shape, dtype=dtype) for data in rgb]).transpose(
            [1, 2, 0]
        )
        if as_float:
            rgb = rgb / np.iinfo(rgb.dtype).max
            rgb.astype(np.float32)
        return rgb

    def plot_fov(
        self,
        fov,
        plot_boundary=True,
        plot_cell_centers=True,
        genes=None,
        image_names=("DAPI+PolyT", "DAPI", "PolyT"),
        padding=150,
        ax_width=5,
        dpi=300,
        n_cols=3,
        hue_range=0.9,
    ):
        """
        Plot fov DAPI + PolyT and other smFISH images (if exists and provided) with transcript spots overlay.

        Parameters
        ----------
        fov :
            Fov number to plot
        plot_boundary :
            whether to plot cell boundaries
        plot_cell_centers :
            whether to plot cell centers
        genes :
            List of genes to plot their transcripts.
        image_names :
            List of image names to plot. See self.image_names for available names.
            Note that the "DAPI+PolyT" is a special name, which means plotting DAPI
            in blue chanel and PolyT in red chanel of the same plot.
        padding :
            Padding in pixels to add to the image extent.
        ax_width :
            Width of the axes in inches.
        dpi :
            DPI of the plot.
        n_cols :
            Number of columns in the plot.
        hue_range :
            A float between 0 and 1. The color range of the image. vmax = vmin + contrast * (vmax - vmin).

        Returns
        -------
        fig :
            Figure object.
        """
        # check parameters
        if genes is None:
            genes = []
        if isinstance(image_names, str):
            image_names = [image_names]
        for name in image_names:
            if name == "DAPI+PolyT":
                continue
            else:
                if name not in self.image_names:
                    raise ValueError(f"{name} not found")

        # Prepare data
        cell_meta = self.get_cell_metadata(fov=fov)
        boundaries = self.get_cell_boundaries(fov=fov)
        transcripts = self.get_transcripts(fov)
        xmin, ymin, xmax, ymax = self.get_fov_pixel_extent_from_transcripts(fov, padding=padding)
        offset = (xmin, ymin)

        # load gray image data
        fov_images = {
            name: self.get_image_fov(name=name, fov=fov, projection="max", padding=padding, contrast=True)
            for name in image_names
            if name != "DAPI+PolyT"
        }
        # load DAPI+PolyT as RGB image
        if "DAPI+PolyT" in image_names:
            fov_images["DAPI+PolyT"] = self.get_rgb_image(
                b_name="DAPI", r_name="PolyT", as_float=True, fov=fov, projection="max", padding=padding, contrast=True
            )

        # make plots
        n_images = len(image_names)
        n_rows = int(np.ceil(n_images / n_cols))
        fig = plt.figure(figsize=(n_cols * ax_width, n_rows * ax_width), dpi=dpi)
        fig.suptitle(f"fov {fov}\n{cell_meta.shape[0]} cells\n{transcripts.shape[0]} transcripts")
        gs = fig.add_gridspec(n_rows, n_cols)

        def _plot(_ax, _image, _cmap):
            plotter = MerfishImageAxesPlotter(
                ax=_ax,
                image=_image,
                boundaries=boundaries,
                cells=cell_meta[["center_x", "center_y"]],
                transform=self.transform,
                offset=offset,
            )
            plotter.plot_image(cmap=_cmap, hue_range=hue_range)
            if plot_boundary:
                plotter.plot_boundaries()
            if plot_cell_centers:
                plotter.plot_cell_centers(s=10)
            for gene in genes:
                gene_data = transcripts.loc[transcripts["gene"] == gene, ["global_x", "global_y"]]
                plotter.plot_scatters(gene_data, label=gene, s=0.5, linewidth=0)

        # plot default nuclei and cytoplasma images
        default_images = {"DAPI+PolyT": None, "DAPI": "Blues", "PolyT": "Reds"}
        plot_i = 0
        for default_image, cmap in default_images.items():
            if default_image in fov_images:
                row = int(np.floor(plot_i / n_cols))
                col = plot_i % n_cols
                image = fov_images.pop(default_image)
                ax = fig.add_subplot(gs[row, col])
                _plot(ax, image, cmap)
                ax.set_title(default_image)
                plot_i += 1

        # plot other images
        for name, gene_image in fov_images.items():
            row = int(np.floor(plot_i / n_cols))
            col = plot_i % n_cols
            ax = fig.add_subplot(gs[row, col])
            _plot(ax, gene_image, "viridis")
            ax.set_title(name)
            plot_i += 1
        return fig
