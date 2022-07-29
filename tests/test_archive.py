import os
import pathlib
import shutil

from merfishing.core.archive_data import ArchiveMerfishExperiment


def test_archive():
    # prepare test file
    raw_path = pathlib.Path("tests/test_data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)
    output_path = pathlib.Path("tests/test_data/output")
    output_path.mkdir(parents=True, exist_ok=True)

    # archive
    archive = ArchiveMerfishExperiment("tests/test_data")
    archive.prepare_archive()
    assert pathlib.Path("tests/test_data/test_data.tar.gz").exists()

    # cleanup test file
    shutil.rmtree("tests/test_data/raw")
    shutil.rmtree("tests/test_data/output")
    os.unlink("tests/test_data/test_data.tar.gz")
    return
