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

    def __init__(self, experiment_dir, verbose=True, *args, **kwargs):
        experiment_dir = pathlib.Path(experiment_dir).absolute()
        if verbose:
            print(f"Experiment dir located at {experiment_dir}")
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
        self._codebook = None
        self._experiment_info = None
        self._fov_positions = None

        # region_dirs
        self.region_dirs = [pathlib.Path(p) for p in self.output_dir.glob("region*")]
        return

    @property
    def experiment_info(self):
        """Get experiment information."""
        if self._experiment_info is None:
            try:
                self._experiment_info = _json_to_namedtuple(self.raw_dir / "experiment.json")
            except FileNotFoundError:
                raise ValueError(f"{self.raw_dir} does not contain an experiment.json file")
        return self._experiment_info

    @property
    def codebook(self):
        """Get codebook."""
        if self._codebook is None:
            try:
                codebook_path = list(self.raw_dir.glob("codebook*.csv"))[0]
            except IndexError:
                raise ValueError(f"{self.raw_dir} does not contain a codebook")
            df = pd.read_csv(codebook_path, index_col=0)
            df["barcode_id"] = range(df.shape[0])
            self._codebook = df
        return self._codebook

    @property
    def fov_positions(self):
        """Get fov positions."""
        if self._fov_positions is None:
            try:
                df = pd.read_csv(self.raw_dir / "settings/positions.csv", index_col=None, header=None)
            except FileNotFoundError:
                raise ValueError(f"{self.raw_dir} does not contain a settings/positions.csv file")
            df.index.name = "fov_id"
            df.columns = ["x", "y"]
            self._fov_positions = df
        return self._fov_positions


class MerfishRegionDirStructureMixin(MerfishExperimentDirStructureMixin):
    """Deal with single MERFISH region level data paths."""

    def __init__(self, region_dir, verbose=True, cell_segmentation="cellpose"):

        region_dir = pathlib.Path(region_dir).absolute()
        assert region_dir.exists(), f"{region_dir} does not exist"
        if verbose:
            print(f"Region data located at {region_dir}")

        # init experiment level information
        experiment_dir = region_dir.parent.parent
        super().__init__(experiment_dir, verbose=verbose)

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

        # cellpose cell segmentation
        self._cellpose_cell_metadata_path = self.region_dir / "cell_metadata.cellpose.csv.gz"
        self._cellpose_cell_by_gene_path = self.region_dir / "cell_by_gene.cellpose.csv.gz"
        self._cellpose_cell_mask_path = self.region_dir / "cell_masks.cellpose"
        complete_flag = all(
            [
                self._cellpose_cell_metadata_path.exists(),
                self._cellpose_cell_by_gene_path.exists(),
                self._cellpose_cell_mask_path.exists(),
            ]
        )
        if not complete_flag:
            self._cellpose_cell_metadata_path = None
            self._cellpose_cell_by_gene_path = None
            self._cellpose_cell_mask_path = None

        # additional analysis output
        self._smfish_transcripts_path_default = self.images_dir / "smfish_transcripts.hdf5"
        if self._smfish_transcripts_path_default.exists():
            self.smfish_transcripts_path = self._smfish_transcripts_path_default
        else:
            # if there is no smFISH data or smFISH spot has not been generated, this will be None
            self.smfish_transcripts_path = None

        # use cell info
        self.segmentation_method = cell_segmentation
        if self.segmentation_method == "cellpose":
            if complete_flag:
                if verbose:
                    print("Using cellpose results")
            else:
                if verbose:
                    print(
                        f"{self.region_dir} does not contain cellpose results or the results are incomplete, "
                        f"using watershed results from vizgen pipeline instead."
                    )
                self.segmentation_method = "watershed"
        if self.segmentation_method == "watershed":
            self.cell_metadata_path = self._watershed_cell_metadata_path
            self.cell_by_gene_path = self._watershed_cell_by_gene_path
        else:
            self.cell_metadata_path = self._cellpose_cell_metadata_path
            self.cell_by_gene_path = self._cellpose_cell_by_gene_path
        return
