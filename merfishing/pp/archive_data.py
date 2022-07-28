import pathlib
import re
import shutil
import subprocess
import warnings
from collections import defaultdict

import pandas as pd
import xarray as xr
from tifffile import imread


def _tif_to_zarr(tif_path, chunk_size=10000):
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


class ArchiveMerfishExperiment:
    """Archive MERFISH raw and output directories."""

    def __init__(self, experiment_dir):
        experiment_dir = pathlib.Path(experiment_dir).absolute()
        assert experiment_dir.exists(), f"{experiment_dir} does not exist"
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_dir.name

        self.raw_path = experiment_dir / "raw"
        assert self.raw_path.exists(), f"{self.raw_path} does not exist, please put the raw data in this directory"

        self.output_path = experiment_dir / "output"
        assert self.output_path.exists(), (
            f"{self.output_path} does not exist, " f"please put the output directory in this directory"
        )

        # execute the archive process
        self.prepare_archive()
        return

    def _convert_tif_to_zarr(self):
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

    def _save_transcripts_to_hdf(self):
        """Save transcripts to HDF5 file."""
        for path in self.output_path.glob("*/detected_transcripts.txt"):
            output_path = path.parent / "detected_transcripts.hdf5"
            transcripts = pd.read_csv(
                path,
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

            # delete input path
            path.unlink()
        return

    def _delete_raw_dir_data(self):
        """Delete the raw path directory and files inside."""
        shutil.rmtree(self.raw_path / "data", ignore_errors=True)
        shutil.rmtree(self.raw_path / "low_resolution", ignore_errors=True)
        shutil.rmtree(self.raw_path / "seg_preview", ignore_errors=True)
        return

    def _compress_vizgen_output(self):
        """Compress the vizgen default output csv files."""
        for path in self.output_path.glob("*/*.csv"):
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

    def prepare_archive(self):
        """Prepare the archive."""
        tar_path = self.experiment_dir / f"{self.experiment_name}.tar.gz"
        _tar_dir([self.raw_path, self.output_path], tar_path)
        print(f"Archive Raw and Output Data: {tar_path}")

        self._convert_tif_to_zarr()
        self._save_transcripts_to_hdf()
        self._compress_vizgen_output()
        self._delete_raw_dir_data()
        return
