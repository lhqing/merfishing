import xarray as xr
from bigfish.stack import (
    focus_projection,
    maximum_projection,
    mean_projection,
    median_projection,
    rescale,
)


def project_image_z(image, z_projection):
    """
    Project image along z axis.

    Parameters
    ----------
    image :
        3-D Image to project.
    z_projection :
        Projection type. Possible values are:
        - "max": Maximum projection.
        - "focus": Focus projection.
        - "mean": Mean projection.
        - "median": Median projection.

    Returns
    -------
    2-D image.
    """
    if z_projection[:3] == "max":
        return maximum_projection(image)
    elif z_projection == "focus":
        return focus_projection(image)
    elif z_projection == "mean":
        return mean_projection(image)
    elif z_projection == "median":
        return median_projection(image)
    else:
        raise ValueError(f"{z_projection} is not a valid z projection")


class MerfishMosaicImage:
    """Merfish mosaic image stored in zarr format."""

    def __init__(self, zarr_path, use_threads=None):
        ds = xr.open_zarr(zarr_path)
        self.da_name = list(ds.data_vars.keys())[0]
        self.image = ds[self.da_name]

        # use_threads = False when reading zarr in multiprocessing mode
        # see documentation here: https://zarr.readthedocs.io/en/stable/tutorial.html#configuring-blosc
        self.image.encoding["compressor"].use_threads = use_threads
        return

    def get_image(self, z, y, x, load=True, projection=None, contrast=True):
        """
        Load image at specific location.

        Parameters
        ----------
        z :
            Z slice.
        y :
            Y slice.
        x :
            X slice.
        load :
            If True, load image from zarr.
        projection :
            Project image along z axis. Only valid if load is True and image is 3D.
        contrast :
            If True, adjust contrast. Only valid if load is True.

        Returns
        -------
        image.
        """
        if z is None:
            z = slice(None)
        if x is None:
            x = slice(None)
        if y is None:
            y = slice(None)

        _img = self.image[z, y, x]
        if load:
            _img = _img.values

            if projection is not None:
                if len(_img.shape) == 3:
                    _img = project_image_z(_img, projection)

            if contrast:
                _img = rescale(_img)
            return _img
        else:
            return _img
