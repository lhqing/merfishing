import xarray as xr


class MerfishMosaicImage(xr.Dataset):
    """Merfish mosaic image stored in zarr format."""

    __slots__ = ()

    def __init__(self, zarr_path):
        ds = xr.open_zarr(zarr_path, mode="r")

        super().__init__(data_vars=ds.data_vars, coords=ds.coords, attrs=ds.attrs)
        return

    @property
    def image_name(self):
        """Get image name."""
        return list(self.data_vars.keys())[0]
