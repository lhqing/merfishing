import json
import pathlib
from collections import namedtuple

import pandas as pd


def _json_to_namedtuple(json_path):
    """Convert json string to named tuple."""
    with open(json_path) as f:
        data = json.load(f)
    json_name = pathlib.Path(json_path).stem
    return namedtuple(json_name, data.keys())(*data.values())


class MerfishExperimentDirStructureMixin:
    """Deal with single MERFISH experiment level data paths."""

    def __init__(self, experiment_dir):
        experiment_dir = pathlib.Path(experiment_dir).absolute()
        assert experiment_dir.exists(), f"{experiment_dir} does not exist"
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_dir.name

        self.raw_dir = experiment_dir / "raw"
        assert self.raw_dir.exists(), f"{self.raw_dir} does not exist, please put the raw data in this directory"

        self.output_dir = experiment_dir / "output"
        assert self.output_dir.exists(), (
            f"{self.output_dir} does not exist, " f"please put the output directory in this directory"
        )

        # experiment information
        self.codebook_path = list(self.raw_dir.glob("codebook*.csv"))[0]
        self.codebook = self._read_codebook()
        self.experiment_info = _json_to_namedtuple(self.experiment_dir / "experiment.json")
        self.fov_positions = self._read_fov_positions()

        # region_dirs
        self.region_dirs = [pathlib.Path(p) for p in self.output_dir.glob("region*")]
        return

    def _read_codebook(self):
        df = pd.read_csv(self.codebook_path, index_col=0)
        df["barcode_id"] = range(df.shape[0])
        return df

    def _read_fov_positions(self):
        df = pd.read_csv(self.experiment_dir / "settings/positions.csv", index_col=None, header=None)
        df.index.name = "fov_id"
        df.columns = ["x", "y"]
        return df


class MerfishRegionDirStructureMixin(MerfishExperimentDirStructureMixin):
    """Deal with single MERFISH region level data paths."""

    def __init__self(self, region_dir):
        region_dir = pathlib.Path(region_dir).absolute()
        assert region_dir.exists(), f"{region_dir} does not exist"

        # init experiment level information
        experiment_dir = region_dir.parent.parent
        super().__init__(experiment_dir)

        # region dir
        region_dir = pathlib.Path(region_dir).absolute()
        self.region_dir = region_dir
        self.region_name = self.region_dir.name
        self.images_dir = self.region_dir / "images"
        self._mosaic_image_zarr_paths = {p.name.split(".")[0].split("_")[-1]: p for p in self.images_dir.glob("*.zarr")}

        # image manifest
        self.image_manifest = _json_to_namedtuple(self.images_dir / "manifest.json")
        self.micron_to_pixel_transform_path = self.images_dir / "micron_to_mosaic_pixel_transform.csv"

        # vizgen default output
        self.transcripts_path = self.region_dir / "detected_transcripts.hdf5"
        self.cell_boundary_dir = self.region_dir / "cell_boundaries"
        # watershed cell segmentation {fov int id: fov hdf5 path}
        self._watershed_cell_boundary_hdf_paths = {
            int(p.name.split(".")[0].split("_")[-1]): p for p in pathlib.Path(self.cell_boundary_dir).glob("*.hdf5")
        }
        self._watershed_cell_metadata_path = self.region_dir / "cell_metadata.csv.gz"
        self._watershed_cell_by_gene_path = self.region_dir / "cell_by_gene.csv.gz"

        # use cell info
        # TODO change to cellpose2 results
        self.cell_metadata_path = self._watershed_cell_metadata_path
        self.cell_by_gene_path = self._watershed_cell_by_gene_path
        return
