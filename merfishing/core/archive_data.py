import pathlib
import re
import shutil
import subprocess
import warnings
from collections import defaultdict

import pandas as pd
import xarray as xr
from tifffile import imread

from .dataset import MerfishExperimentDirStructureMixin, MerfishRegionDirStructureMixin


def _tif_to_zarr(tif_path, chunk_size=5000):
    img = imread(str(tif_path))
    # tiff image is (y, x), append dims add z in the first dim
    # so the da dim is (z, y, x)
    da = xr.DataArray(img, dims=["y", "x"]).expand_dims("z")
    da.encoding["chunks"] = (1, chunk_size, chunk_size)

    tif_path = pathlib.Path(tif_path)
    name_prefix = "_".join(tif_path.name.split("_")[:-1])
    ds = xr.Dataset({name_prefix: da})

    output_path = tif_path.parent / f"{name_prefix}.zarr"
    if output_path.exists():
        ds.to_zarr(output_path, append_dim="z")
    else:
        ds.to_zarr(output_path)

    # improve zarr chunking
    _rechunk_tiff_zarr(output_path, name_prefix, chunk_size)
    return


def _rechunk_tiff_zarr(image_path, image_name, chunk_size=5000):
    """Rechunk tiff zarr by combine the z dim and chunk on x and y dims."""
    temp_path = f"{pathlib.Path(image_path).name}.temp"
    # make sure temp path do not exist
    shutil.rmtree(temp_path, ignore_errors=True)

    image = xr.open_zarr(image_path)

    image = image.rename({"dim_0": "y", "dim_1": "x"})

    image[image_name].encoding["chunks"] = (image.dims["z"], chunk_size, chunk_size)
    image[image_name].encoding["preferred_chunks"] = {"z": image.dims["z"], "y": chunk_size, "x": chunk_size}

    # creates metadata only, not actually writing
    image.to_zarr(temp_path, compute=False, safe_chunks=False, mode="w")
    x_size = image.dims["x"]
    y_size = image.dims["y"]

    for x_start in range(0, x_size, chunk_size):
        x_slice = slice(x_start, x_start + chunk_size)
        for y_start in range(0, y_size, chunk_size):
            y_slice = slice(y_start, y_start + chunk_size)
            image_chunk = image.isel(z=slice(None), x=x_slice, y=y_slice)
            image_chunk.load()
            image_chunk.to_zarr(temp_path, region={"x": x_slice, "y": y_slice})

    # swap the new zarr with the old one
    image_path = pathlib.Path(image_path)
    image_path = image_path.parent / image_path.name
    shutil.move(image_path, f"{image_path}.to_be_deleted")
    shutil.move(temp_path, image_path)
    shutil.rmtree(f"{image_path}.to_be_deleted", ignore_errors=True)
    return


def _tar_dir(dir_paths, tar_path):
    dir_paths = " ".join(map(str, dir_paths))
    try:
        subprocess.run(
            f"tar -cf {tar_path} --use-compress-program=pigz {dir_paths}",
            shell=True,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        try:
            print("tar failed with pigz, trying gzip again...")
            print(
                'If pigz is not installed, please run "mamba install pigz" to install it. '
                "It will be much faster than gzip."
            )
            subprocess.run(
                f"tar -cf {tar_path} {dir_paths}",
                shell=True,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode())
            print(e.stderr.decode())
            raise e
    return tar_path


class ArchiveMerfishRegion(MerfishRegionDirStructureMixin):
    """Archive MERFISH raw and output directories."""

    def __init__(self, region_dir):
        super(MerfishRegionDirStructureMixin, self).__init__(region_dir)

        # execute the archive process
        self.prepare_archive()
        return

    def _convert_tif_to_zarr(self):
        # turn all image TIF files into zarr files
        p = re.compile(r"\S+_(?P<name>\S+)_z(?P<zorder>\d+).tif")

        # get all tif files
        tif_dict = defaultdict(dict)
        for tif_path in pathlib.Path(self.images_dir).glob("*.tif"):
            name_dict = p.match(tif_path.name).groupdict()
            zorder = int(name_dict["zorder"])
            name = name_dict["name"]
            tif_dict[name][zorder] = tif_path

        # save each TIF image as zarr, z-stacks are saved together
        for _, zdict in tif_dict.items():
            for _, path in sorted(zdict.items(), key=lambda x: x[0]):
                _tif_to_zarr(path)
                path.unlink()  # delete after successful conversion
        return

    def _compress_vizgen_output(self):
        """Compress the vizgen default output csv files."""
        for path in self.region_dir.glob("*.csv"):
            try:
                subprocess.run(
                    f"pigz {path}",
                    shell=True,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(e.stdout.decode())
                print(e.stderr.decode())
                raise e
        return

    def _save_transcripts_to_hdf(self):
        """Save transcripts to HDF5 file."""
        output_path = self.transcripts_path

        transcripts = pd.read_csv(
            self.region_dir / "detected_transcripts.csv",
            index_col=0,
            dtype={
                "barcode_id": "uint16",
                "global_x": "float32",
                "global_y": "float32",
                "global_z": "uint16",
                "x": "float32",
                "y": "float32",
                "fov": "uint16",
                "gene": "str",
                "transcript_id": "str",
            },
        ).sort_values("fov")

        with pd.HDFStore(str(output_path), complevel=1, complib="blosc") as hdf:
            for fov, fov_df in transcripts.groupby("fov"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hdf[str(fov)] = fov_df
        return

    def prepare_archive(self):
        """Prepare the region archive."""
        self._convert_tif_to_zarr()
        print(f"{self.region_name}: Converted TIF files to Zarr")

        self._save_transcripts_to_hdf()
        print(f"{self.region_name}: Saved transcripts to HDF5")

        self._compress_vizgen_output()
        print(f"{self.region_name}: Compressed vizgen output")
        return


class ArchiveMerfishExperiment(MerfishExperimentDirStructureMixin):
    """Archive MERFISH raw and output directories."""

    def __init__(self, experiment_dir):
        super(MerfishExperimentDirStructureMixin, self).__init__(experiment_dir)

        # execute the archive process
        self.prepare_archive()
        return

    def _delete_raw_dir_data(self):
        """Delete the raw path directory and files inside."""
        shutil.rmtree(self.raw_dir / "data", ignore_errors=True)
        shutil.rmtree(self.raw_dir / "low_resolution", ignore_errors=True)
        shutil.rmtree(self.raw_dir / "seg_preview", ignore_errors=True)
        return

    def prepare_archive(self):
        """Prepare the experiment archive."""
        tar_path = self.experiment_dir / f"{self.experiment_name}.tar.gz"
        _tar_dir([self.raw_dir, self.output_dir], tar_path)
        print(f"{self.experiment_name}: Archive Raw and Output Data: {tar_path}")

        for region_dir in self.region_dirs:
            ArchiveMerfishRegion(region_dir)

        self._delete_raw_dir_data()
        print(f"{self.experiment_name}: Deleted raw data")
        return
