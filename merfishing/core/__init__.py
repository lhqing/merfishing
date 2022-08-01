import pathlib
import shutil
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..pl.plot_image import MerfishImageAxesPlotter
from .boundary import WatershedCellBoundary
from .dataset import MerfishRegionDirStructureMixin
from .image import MerfishMosaicImage
from .transform import MerfishTransform


def _call_spots_single_fov(
    image_path, x, y, output_path, plot=False, detect_dense=True, projection="max", verbose=False, **spot_kwargs
):
    """Worker function for calling spots in a single image slice and save data on disk."""
    from ..tl.smfish import call_spot

    # skip if output file exists
    success_path = output_path.with_suffix(".success")
    if success_path.exists():
        return

    if verbose:
        print(f"Calling spots in {pathlib.Path(image_path).name}, x={x}, y={y}," f"\nsaving to {output_path}")

    _image = MerfishMosaicImage(image_path)
    fov_image = _image.get_image(z=None, y=y, x=x, load=True, contrast=False, projection=projection)

    result, *_ = call_spot(fov_image, detect_dense=detect_dense, plot=plot, verbose=verbose, **spot_kwargs)

    # save result to temp npy file
    np.save(output_path, result)
    # touch a success file flag
    with open(success_path, "w") as _:
        pass
    return


class MerfishExperimentRegion(MerfishRegionDirStructureMixin):
    """Entry point for one region of a MERFISH experiment."""

    def __init__(self, region_dir):
        print("MERFISH Experiment Region")
        # setup all the file paths and load experiment manifest information
        super().__init__(region_dir)

        # Transform image coords between micron and pixel
        self.transform = MerfishTransform(self.micron_to_pixel_transform_path)

        # cell metadata
        self._cell_metadata = None

        # fov
        self._fov_ids = None
        self._fov_micron_extends = {}
        return

    @property
    def image_names(self):
        """Get image names."""
        return list(self._mosaic_image_zarr_paths.keys())

    @property
    def smfish_genes(self):
        """Get smfish genes."""
        return [g for g in self._mosaic_image_zarr_paths.keys() if g not in ["DAPI", "PolyT"]]

    # ==========================================================================
    # Get information
    # ==========================================================================

    def get_cell_metadata(self, fov=None):
        """Get cell metadata."""
        if self._cell_metadata is None:
            df = pd.read_csv(self.cell_metadata_path, index_col=0)
            self._cell_metadata = df
        df = self._cell_metadata

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

    def get_transcripts(self, fov, smfish=True):
        """Get transcripts detected in the FOV."""
        df = pd.read_hdf(self.transcripts_path, key=str(fov))

        # if there are smFISH transcripts, add them to the df
        if smfish and self.smfish_transcripts_path is not None:
            fish_df = pd.read_hdf(self.smfish_transcripts_path, key=str(fov))
            # noinspection PyTypeChecker
            df = pd.concat([df, fish_df])
        return df

    @property
    def fov_ids(self):
        """Get fov ids."""
        if self._fov_ids is None:
            boundary_paths = list(self.cell_boundary_dir.glob("feature_data_*.hdf5"))
            self._fov_ids = [pathlib.Path(p).stem.split("_")[-1] for p in boundary_paths]
            self._fov_ids = sorted(set(self._fov_ids))
        return self._fov_ids

    # ==========================================================================
    # Get mosaic image, deal with FOV coords and make plots
    # ==========================================================================

    def get_image(self, name):
        """Get image by name, and select fov (optional) and z slice (optional)."""
        try:
            zarr_path = self._mosaic_image_zarr_paths[name]
        except KeyError:
            raise KeyError(f"Do not have {name} image, " f"possible names are {self.image_names}")
        img = MerfishMosaicImage(zarr_path)
        return img

    def get_rgb_image(self, name=None, r_name=None, g_name=None, b_name=None, as_float=False, **kwargs):
        """
        Get RGB image from up to three different mosaic images.

        Parameters
        ----------
        name : str
            RGB name separated by "+"
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

        if name is not None:
            r_name, g_name, b_name = name.split("+")
            if r_name == "":
                r_name = None
            if g_name == "":
                g_name = None
            if b_name == "":
                b_name = None

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

    def get_fov_micron_extent_from_transcripts(self, fov):
        """
        Get fov extent in micron coords from transcripts detected in the FOV.

        Parameters
        ----------
        fov :
            fov id

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            fov extent in micron coords
        """
        # cache the results to prevent repeated transcripts lookup
        if fov not in self._fov_micron_extends:
            transcripts = self.get_transcripts(fov, smfish=False)
            xmin, ymin = transcripts[["global_x", "global_y"]].min()
            xmax, ymax = transcripts[["global_x", "global_y"]].max()
            self._fov_micron_extends[fov] = (xmin, ymin, xmax, ymax)
        else:
            xmin, ymin, xmax, ymax = self._fov_micron_extends[fov]
        return xmin, ymin, xmax, ymax

    def get_fov_pixel_extent_from_transcripts(self, fov, padding=0):
        """
        Get fov extent in pixel coords from transcripts detected in the FOV.

        Parameters
        ----------
        fov :
            fov id
        padding :
            padding in pixels

        Returns
        -------
        xmin, ymin, xmax, ymax : int
            fov extent in pixel coords
        """
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

    def plot_fov(
        self,
        fov,
        plot_boundary=True,
        plot_cell_centers=True,
        genes=None,
        image_names=("PolyT++DAPI", "DAPI", "PolyT"),
        padding=150,
        ax_width=5,
        dpi=300,
        n_cols=3,
        hue_range=0.9,
    ):
        """
        Plot fov PolyT + DAPI and other smFISH images (if exists and provided) with transcript spots overlay.

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
            RGB image can be specified by name separated by "R+G+B", missing chanel will be filled with zeros.
            For example, the "PolyT++DAPI" is a RGB name, which means plotting
            PolyT in red chanel (1) and DAPI in blue chanel (3).
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
            if "+" in name:
                for _name in name.split("+"):
                    if _name == "":
                        continue
                    if _name not in self.image_names:
                        raise ValueError(f"{_name} not found")
            else:
                if name not in self.image_names:
                    raise ValueError(f"{name} not found")

        # Prepare data
        cell_meta = self.get_cell_metadata(fov=fov)
        boundaries = self.get_cell_boundaries(fov=fov)
        transcripts = self.get_transcripts(fov, smfish=True)
        xmin, ymin, xmax, ymax = self.get_fov_pixel_extent_from_transcripts(fov, padding=padding)
        offset = (xmin, ymin)

        # load gray image data
        fov_images = {
            name: self.get_image_fov(name=name, fov=fov, projection="max", padding=padding, contrast=True)
            if "+" not in name
            else self.get_rgb_image(name=name, as_float=True, fov=fov, projection="max", padding=padding, contrast=True)
            for name in image_names
        }

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
        default_image_cmap = {"DAPI": "Blues", "PolyT": "Reds"}
        plot_i = 0
        for name in image_names:
            gene_image = fov_images[name]
            if name in default_image_cmap:
                cmap = default_image_cmap[name]
            else:
                if "+" in name:
                    cmap = None
                else:
                    cmap = "viridis"

            row = int(np.floor(plot_i / n_cols))
            col = plot_i % n_cols

            ax = fig.add_subplot(gs[row, col])
            _plot(ax, gene_image, cmap)
            ax.set_title(name)
            plot_i += 1
        return fig

    # ==========================================================================
    # smFISH analysis
    # ==========================================================================

    def _convert_spots_results(self, result_path, offset, image_name, fov):
        """Convert call_spots result to transcript format as vizgen output."""
        # load npy
        result = np.load(result_path)

        # deal with coordinates
        local_spot_pixel = pd.DataFrame(result, columns=["y", "x"])
        global_spot_pixel = local_spot_pixel.copy()
        xmin, ymin = offset
        global_spot_pixel["x"] += xmin
        global_spot_pixel["y"] += ymin

        # IMPORTANT: transform function take coordinates with order (x, y) and return (x, y)
        local_spot_micron = self.transform.pixel_to_micron_transform(local_spot_pixel[["x", "y"]])
        global_spot_micron = self.transform.pixel_to_micron_transform(global_spot_pixel[["x", "y"]])

        local_spot_micron = pd.DataFrame(local_spot_micron, columns=["x", "y"])
        global_spot_micron = pd.DataFrame(global_spot_micron, columns=["global_x", "global_y"])

        transcript = pd.DataFrame(
            {
                "barcode_id": self.codebook.loc[image_name, "barcode_id"],
                "transcript_id": self.codebook.loc[image_name, "id"],
                "gene": image_name,
                "global_z": result[:, 0],
                "fov": fov,
            }
        )

        columns = ["barcode_id", "global_x", "global_y", "global_z", "x", "y", "fov", "gene", "transcript_id"]
        transcript = pd.concat([transcript, global_spot_micron, local_spot_micron], axis=1).reindex(columns=columns)
        return transcript

    def call_spots(
        self,
        image_names="all",
        cpu=1,
        padding=50,
        detect_dense=True,
        projection="max",
        verbose=False,
        redo=False,
        **spot_kwargs,
    ):
        """
        Call spots on all smFISH images.

        Parameters
        ----------
        image_names :
            Image names to call spots on. If 'all', call spots on all images with self.image_names.
        cpu :
            Number of CPUs to use.
        padding :
            Padding to add to image borders.
        detect_dense :
            Whether to detect dense spots using the bigfish.detection.detect_dense function.
        projection :
            Projection to use for project z axis.
        verbose :
            Whether to print progress.
        redo :
            Whether to redo the analysis when the smFISH results already exist.
        spot_kwargs :
            Keyword arguments to pass to merfishing.tl.call_spots.
        """
        if self.smfish_transcripts_path is not None:
            if redo:
                print("smFISH transcripts already exist, but redo is True. Deleting...")
                pathlib.Path(self.smfish_transcripts_path).unlink()
            else:
                print(
                    f"smFISH transcripts already exist at {self.smfish_transcripts_path}. "
                    f"Use redo=True to redo the analysis."
                )
                return

        temp_dir = self.region_dir / "smFISH_spot_temp"
        temp_dir.mkdir(exist_ok=True)

        if image_names == "all":
            image_names = self.smfish_genes
        elif isinstance(image_names, str):
            image_names = [image_names]
        else:
            pass
        for name in image_names:
            if name not in self.smfish_genes:
                raise ValueError(f"{name} is not a valid image name, " f"available images are {self.smfish_genes}")

        with ProcessPoolExecutor(cpu) as executor:
            futures = {}
            for image_name in image_names:
                image_path = self._mosaic_image_zarr_paths[image_name]
                for fov in self.fov_ids:
                    xmin, ymin, xmax, ymax = self.get_fov_pixel_extent_from_transcripts(fov=fov, padding=padding)
                    x_slice = slice(xmin, xmax)
                    y_slice = slice(ymin, ymax)
                    output_path = temp_dir / f"{image_name}_{fov}.npy"
                    future = executor.submit(
                        _call_spots_single_fov,
                        image_path=image_path,
                        x=x_slice,
                        y=y_slice,
                        output_path=output_path,
                        plot=False,
                        detect_dense=detect_dense,
                        projection=projection,
                        verbose=verbose,
                        **spot_kwargs,
                    )
                    futures[future] = (image_name, fov, output_path)

            for future in as_completed(futures):
                image_name, fov, output_path = futures[future]
                if verbose:
                    print(f"{image_name} {fov} finished")
                future.result()

        # merge all results into single pd.HDFStore
        # self._smfish_transcripts_path_default is the default name to save smFISH transcripts
        temp_image_hdf_path = str(self._smfish_transcripts_path_default) + ".temp"

        # get all paths
        fov_results = defaultdict(dict)
        for path in temp_dir.glob("*.npy"):
            image_name, fov = path.stem.split("_")
            fov_results[fov][image_name] = path

        # save transcripts per fov
        for fov, fov_images in fov_results.items():
            if verbose:
                print(f"Writing smFISH transcripts for fov {fov}")
            transcripts = []
            for image_name, result_path in fov_images.items():
                xmin, ymin, *_ = self.get_fov_pixel_extent_from_transcripts(fov=fov, padding=padding)
                transcript = self._convert_spots_results(
                    result_path=result_path, offset=(xmin, ymin), image_name=image_name, fov=fov
                )
                transcripts.append(transcript)
            transcripts = pd.concat(transcripts)
            with pd.HDFStore(temp_image_hdf_path, "a") as store:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    store.put(f"/{fov}", transcripts, format="table", data_columns=True, complib="blosc", complevel=3)
        shutil.move(temp_image_hdf_path, self._smfish_transcripts_path_default)

        # the smFISH transcripts are saved in the default path, put the path to the public variable
        # self.get_transcripts() will be able to load the transcripts together with the MERFISH transcripts
        self.smfish_transcripts_path = self._smfish_transcripts_path_default

        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # ==========================================================================
    # Cell segmentation analysis
    # ==========================================================================
