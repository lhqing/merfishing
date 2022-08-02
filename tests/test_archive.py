import os

from merfishing.core.archive_data import ArchiveMerfishExperiment

from ._utils import DUMMY_EXPERIMENT_PATH


def test_archive():
    # archive
    ArchiveMerfishExperiment(DUMMY_EXPERIMENT_PATH)

    # cleanup test file
    os.unlink(DUMMY_EXPERIMENT_PATH / "dummy_experiment.tar.gz")
    return
