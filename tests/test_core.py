import shutil

from merfishing.core import MerfishExperimentRegion

from ._utils import DUMMY_REGION_PATH


def _check_basics(merfish):
    image_names = merfish.image_names
    smfish_names = merfish.smfish_genes
    assert "DAPI" in image_names
    assert "PolyT" in image_names
    assert "DAPI" not in smfish_names
    assert "PolyT" not in smfish_names

    merfish.call_spots(cpu=8, verbose=False, redo=True)

    fov_ids = merfish.fov_ids
    fov = fov_ids[0]
    meta = merfish.get_cell_metadata()
    assert meta.shape[0] > 0
    meta = merfish.get_cell_metadata(fov=fov)
    print(fov, type(fov))
    assert meta.shape[0] > 0

    boundaries = merfish.get_cell_boundaries(fov=fov)
    merfish.get_cell_boundaries(fov=fov, cells=list(boundaries.keys())[:1])
    assert len(boundaries) > 0

    transcripts = merfish.get_transcripts(fov=fov, smfish=True)
    for g in merfish.smfish_genes:
        assert g in transcripts["gene"].unique()

    merfish.plot_fov(fov, padding=30)

    # cleanup
    assert merfish.smfish_transcripts_path.exists()
    # merfish.smfish_transcripts_path.unlink(missing_ok=True)


def test_watershed_merfish():
    merfish = MerfishExperimentRegion(DUMMY_REGION_PATH, cell_segmentation="watershed")
    _check_basics(merfish)


def test_cellpose_merfish():
    merfish = MerfishExperimentRegion(DUMMY_REGION_PATH)
    merfish.cell_segmentation(model_type="cyto", diameter=100, jobs=8, verbose=False, redo=True)
    merfish = MerfishExperimentRegion(DUMMY_REGION_PATH, cell_segmentation="cellpose")
    assert merfish.segmentation_method == "cellpose"

    _check_basics(merfish)

    # cleanup
    cell_by_gene = merfish.region_dir / "cell_by_gene.cellpose.csv.gz"
    assert cell_by_gene.exists()
    cell_by_gene.unlink()

    cell_metadata = merfish.region_dir / "cell_metadata.cellpose.csv.gz"
    assert cell_metadata.exists()
    cell_metadata.unlink()

    cell_masks = merfish.region_dir / "cell_masks.cellpose"
    assert cell_masks.exists()
    shutil.rmtree(cell_masks)
