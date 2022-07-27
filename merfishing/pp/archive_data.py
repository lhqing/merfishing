import os
import pathlib
import re
import shutil
import tarfile
from collections import defaultdict

import xarray as xr
from tifffile import imread


def _tif_to_zarr(tif_path, chunk_size=25000):
    img = imread(str(tif_path))
    da = xr.DataArray(img).expand_dims("z")
    da.encoding["chunks"] = (1, chunk_size, chunk_size)

    tif_path = pathlib.Path(tif_path)
    name_prefix = "_".join(tif_path.name.split("_")[:-1])
    ds = xr.Dataset({name_prefix: da})

    output_path = tif_path.parent / f"{name_prefix}.zarr"
    if output_path.exists():
        ds.to_zarr(output_path, append_dim="z")
    else:
        ds.to_zarr(output_path)
    return


def _tar_dir(dir_path):
    dir_path = str(dir_path)
    tar_path = dir_path + ".tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))
    return tar_path


class ArchiveMerfishExperiment:
    """Archive MERFISH raw and output directories."""

    def __init__(self, experiment_dir):
        experiment_dir = pathlib.Path(experiment_dir).absolute()
        assert experiment_dir.exists(), f"{experiment_dir} does not exist"

        self.raw_path = experiment_dir / "raw"
        assert self.raw_path.exists(), f"{self.raw_path} does not exist, please put the raw data in this directory"

        self.output_path = experiment_dir / "output"
        assert self.output_path.exists(), (
            f"{self.output_path} does not exist, " f"please put the output directory in this directory"
        )
        return

    def prepare_archive(self):
        """Prepare the archive."""
        tar_path = _tar_dir(self.raw_path)
        print(f"Archive Raw Data: {tar_path}")
        tar_path = _tar_dir(self.output_path)
        print(f"Archive Output Data: {tar_path}")

        # turn all image TIF files into zarr files
        p = re.compile(r"\S+_(?P<name>\S+)_z(?P<zorder>\d+).tif")

        # get all tif files
        tif_dict = defaultdict(dict)
        for tif_path in pathlib.Path(self.output_path).glob("**/*.tif"):
            name_dict = p.match(tif_path.name).groupdict()
            zorder = int(name_dict["zorder"])
            name = name_dict["name"]
            tif_dict[name][zorder] = tif_path

        # save each TIF image as zarr, z-stacks are saved together
        for _, zdict in tif_dict.items():
            for _, path in sorted(zdict.items(), key=lambda x: x[0]):
                _tif_to_zarr(path)
                path.unlink()  # delete after successful conversion

        self._delete_raw_dir()
        return

    def _delete_raw_dir(self):
        """Delete the raw path directory and files inside."""
        shutil.rmtree(self.raw_path)
        return
