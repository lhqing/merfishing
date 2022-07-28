import h5py
import numpy as np


class CellBoundaryMixin:
    """Common methods for all cell boundaries."""

    def __init__(self):
        self.z_coords = np.array([])
        self._boundaries = {}

    def boundary(self, z):
        """Get boundary for specific z stack."""
        return self._boundaries[int(z)]

    @property
    def boundaries(self):
        """Get ordered boundaries."""
        return [self._boundaries[z] for z in sorted(self._boundaries.keys())]


class WatershedCellBoundary(CellBoundaryMixin):
    """Watershed-algorithm cell boundary, from hdf5 file created by vizgen default analysis."""

    def __init__(self, hdf_path, cell_id):
        super().__init__()

        with h5py.File(hdf_path) as f:
            try:
                cell_group = f[f"featuredata/{cell_id}"]
            except KeyError as e:
                print(f"MERFISH cell {cell_id} not found in {hdf_path}.")
                raise e
            for k, v in cell_group.items():
                if k.startswith("zIndex"):
                    try:
                        self._boundaries[int(k.split("/")[-1].split("_")[-1])] = np.array(v["p_0/coordinates"])
                    except KeyError:
                        # sometimes the boundary might be missing in specific z stack
                        continue
                else:
                    # z coords
                    self.z_coords = np.array(v)
        return
