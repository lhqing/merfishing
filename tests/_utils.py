import pathlib

DUMMY_EXPERIMENT_PATH = pathlib.Path("tests/dummy_experiment").absolute()
DUMMY_REGION_PATH = DUMMY_EXPERIMENT_PATH / "output/region_0"
assert DUMMY_EXPERIMENT_PATH.exists(), f"DUMMY_EXPERIMENT_PATH {DUMMY_EXPERIMENT_PATH} does not exist"
assert DUMMY_REGION_PATH.exists(), f"DUMMY_REGION_PATH {DUMMY_REGION_PATH} does not exist"
