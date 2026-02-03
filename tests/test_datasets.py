import os
from pathlib import Path

import pytest

import pandas as pd

from latent_calendar.datasets import (
    load_chicago_bikes,
    load_online_transactions,
    load_ufo_sightings,
    HERE,
)

DATASETS_DIR = Path(__file__).parents[1] / "datasets"


@pytest.mark.skipif(
    os.environ.get("CI") != "INTEGRATION", reason="CI is not set to INTEGRATION"
)
@pytest.mark.parametrize(
    "load_func, local_file",
    [
        (load_chicago_bikes, "chicago-bikes.csv"),
        (load_online_transactions, "online_retail_II.csv"),
        (load_ufo_sightings, "ufo_sighting_data.csv"),
    ],
)
def test_load_func_local_save(load_func, local_file):
    file: Path = HERE / local_file
    file.unlink(missing_ok=True)

    df = load_func(local_save=True)
    df_second_time = load_func()

    pd.testing.assert_frame_equal(df, df_second_time)


@pytest.mark.parametrize(
    "load_func, local_file",
    [
        (load_chicago_bikes, "chicago-bikes.csv"),
        (load_online_transactions, "online_retail_II.csv"),
        (load_ufo_sightings, "ufo_sighting_data.csv"),
    ],
)
def test_load_func_subset(load_func, local_file: str) -> None:
    actual_file: Path = DATASETS_DIR / local_file
    file: Path = HERE / local_file

    if file.exists():
        file.unlink()

    file.symlink_to(actual_file)

    read_kwargs = {"nrows": 5}
    df = load_func(**read_kwargs)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    # Check for datetime columns (pandas 3.0 uses datetime64[us] instead of datetime64[ns])
    datetime_dtypes = df.dtypes.astype(str).str.startswith("datetime64")
    assert datetime_dtypes.sum() >= 1

    file.unlink()
