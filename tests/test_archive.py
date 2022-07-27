import os
import pathlib
import shutil

from merfishing.pp.archive_data import ArchiveMerfishExperiment


def test_archive():
    # prepare test file
    raw_path = pathlib.Path("tests/test_data/raw")
    raw_path.mkdir(parents=True, exist_ok=True)
    output_path = pathlib.Path("tests/test_data/output")
    output_path.mkdir(parents=True, exist_ok=True)

    # archive
    archive = ArchiveMerfishExperiment(raw_path, output_path)
    archive.prepare_archive()
    assert not raw_path.exists()
    assert pathlib.Path("tests/test_data/raw.tar.gz").exists()
    assert pathlib.Path("tests/test_data/output.tar.gz").exists()

    # cleanup test file
    shutil.rmtree(output_path)
    os.remove(str(output_path) + ".tar.gz")
    os.remove(str(raw_path) + ".tar.gz")
    return
