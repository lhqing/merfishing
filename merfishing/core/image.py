import xarray as xr


class MerfishMosaicImage:
    """Merfish mosaic image stored in zarr format."""

    def __init__(self, zarr_path):
        ds = xr.open_zarr(zarr_path)
        self.image = ds[list(ds.data_vars.keys())[0]]
        return

    def load_image(self, z, y, x):
        """Load image at specific location."""
        return self.image[z, y, x].load()
