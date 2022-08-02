from collections import OrderedDict

import h5py
import numpy as np
import xarray as xr


class CellBoundary:
    """Common methods for all cell boundaries."""

    def __init__(self, z_coords, boundaries):
        self.z_coords: np.ndarray = np.array(sorted(z_coords))
        self._boundaries: OrderedDict = OrderedDict()
        for k in sorted(boundaries.keys()):
            self._boundaries[k] = boundaries[k]

    def boundary(self, z):
        """Get boundary for specific z stack."""
        return self._boundaries[int(z)]

    @property
    def boundaries(self):
        """Get ordered boundaries."""
        return list(self._boundaries.values())

    def __getitem__(self, item):
        return self._boundaries[item]

    def keys(self):
        """Get cell ids."""
        return self._boundaries.keys()

    def values(self):
        """Get boundaries."""
        return self._boundaries.values()

    def items(self):
        """Get cell ids and boundaries."""
        return self._boundaries.items()


def load_watershed_boundaries(hdf_path, cells) -> dict:
    """
    Load watershed boundaries from a hdf file.

    Parameters
    ----------
    hdf_path :
        Path to the hdf file.
    cells :
        List of cell ids.

    Returns
    -------
    A dictionary of cell ids and their boundaries.
    """
    records = {}
    for cell_id in cells:
        with h5py.File(hdf_path) as f:
            try:
                cell_group = f[f"featuredata/{cell_id}"]
            except KeyError as e:
                print(f"MERFISH cell {cell_id} not found in {hdf_path}.")
                raise e
            _boundaries = {}
            for k, v in cell_group.items():
                if k.startswith("zIndex"):
                    try:
                        _boundaries[int(k.split("/")[-1].split("_")[-1])] = np.array(v["p_0/coordinates"])
                    except KeyError:
                        # sometimes the boundary might be missing in specific z stack
                        continue
                else:
                    # z coords
                    z_coords = np.array(v)
        records[cell_id] = CellBoundary(z_coords, _boundaries)
    return records


def load_cellpose_boundaries(mask_path, cells, pixel_to_micron_transform) -> dict:
    """
    Load cellpose boundaries from a mask file.

    Parameters
    ----------
    mask_path :
        Path to the mask zarr file.
    cells :
        List of cell ids.
    pixel_to_micron_transform :
        A function that converts pixel coordinates to micron coordinates.

    Returns
    -------
    A dictionary of cell ids and their boundaries.
    """
    from ..tl.cellpose import NUM_Z_SLICES, Z_SLICE_DISTANCE, outlines_list_3d

    da = xr.open_zarr(mask_path)["mask"]
    mask = da.values
    offset = da.attrs["offset"]

    if cells is not None:
        try:
            cells = np.array(cells).astype(int)
        except ValueError:
            cells = np.array([int(cell.split("_")[-1]) for cell in cells])

    def _transform_outline_pixel_to_micron(coords):
        coords += [[offset[0], offset[1]]]
        return pixel_to_micron_transform(coords)

    outlines = outlines_list_3d(
        mask_3d=mask, feature_ids=cells, transform_func=_transform_outline_pixel_to_micron, as_polygon=False
    )

    records = {}
    for cell_id, z_records in outlines.items():
        z_coords = [Z_SLICE_DISTANCE * k for k in range(1, NUM_Z_SLICES + 1)]
        records[cell_id] = CellBoundary(z_coords, z_records)
    return records
