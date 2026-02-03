"""Example datasets for latent_calendar.

All datasets are loaded from the web and cached locally if desired with the `local_save` argument.

The datasets all include one or more columns that represent a datetime that can be used for calendar analysis.

Examples:
    Load the chicago bikes dataset:

    ```python
    from latent_calendar.datasets import load_chicago_bikes

    df = load_chicago_bikes()
    df.head()
    ```

    ```text
                    start_station_name end_station_name  rideable_type          started_at            ended_at member_casual
    ride_id
    ABF4F851DE485B76                NaN              NaN  electric_bike 2023-06-30 18:56:13 2023-06-30 19:30:40        member
    F123B5D34B002CDB                NaN              NaN  electric_bike 2023-06-30 06:12:31 2023-06-30 06:23:05        member
    CA8E2C38AF641DFB                NaN              NaN  electric_bike 2023-06-30 08:28:51 2023-06-30 08:37:45        member
    93CCE4EA48CFDB69                NaN              NaN  electric_bike 2023-06-30 09:09:24 2023-06-30 09:17:41        member
    FDBCEFE7890F7262                NaN              NaN  electric_bike 2023-06-30 16:29:48 2023-06-30 16:38:51        member
    ```

"""

from pathlib import Path

import pandas as pd

__all__ = ["load_online_transactions", "load_chicago_bikes", "load_ufo_sightings"]


HERE = Path(__file__).parent
BASE_URL = (
    "https://raw.githubusercontent.com/williambdean/latent-calendar/main/datasets"
)


def _download_csv(name: str, **kwargs) -> pd.DataFrame:
    file = f"{BASE_URL}/{name}.csv"

    return pd.read_csv(file, **kwargs)


def _create_local_file_name(name: str) -> Path:
    return HERE / f"{name}.csv"


def _load_data(
    name: str, read_kwargs=None, save_kwargs=None, local_save: bool = False
) -> pd.DataFrame:
    read_kwargs = read_kwargs or {}
    save_kwargs = save_kwargs or {}

    file = _create_local_file_name(name)

    if not file.exists():
        df = _download_csv(name=name, **read_kwargs)
        if local_save:
            df.to_csv(file, **save_kwargs)
    else:
        df = pd.read_csv(file, **read_kwargs)

    return df


def load_online_transactions(local_save: bool = False, **read_kwargs) -> pd.DataFrame:
    """Kaggle Data for an non-store online retailer in UK. More information [here](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).

    Args:
        local_save: Whether to save the data locally if it doesn't exists.
        read_kwargs: kwargs to pass to pd.read_csv

    Returns:
        Online transactions data from a non-store online retailer in UK.

    """
    name = "online_retail_II"
    read_kwargs = {
        "parse_dates": ["InvoiceDate"],
        **read_kwargs,
    }

    return _load_data(name, read_kwargs=read_kwargs, local_save=local_save)


def load_chicago_bikes(local_save: bool = False, **read_kwargs) -> pd.DataFrame:
    """Bikesharing trip level data from Chicago's Divvy system.

    Read more about the data source [here](https://data.cityofchicago.org/Transportation/Divvy-Trips/fg6s-gzvg).

    The data is two weeks of trips starting June 26th, 2023 until July 9th, 2023.

    Args:
        local_save: Whether to save the data locally if it doesn't exists.
        read_kwargs: kwargs to pass to pd.read_csv

    Returns:
        Trips data from Chicago's Divvy system.

    """
    name = "chicago-bikes"
    read_kwargs = {
        "parse_dates": ["started_at", "ended_at"],
        "index_col": ["ride_id"],
        **read_kwargs,
    }

    return _load_data(name, read_kwargs=read_kwargs, local_save=local_save)


def load_ufo_sightings(local_save: bool = False, **read_kwargs) -> pd.DataFrame:
    """UFO sightings over time around the world. More info [here](https://www.kaggle.com/datasets/camnugent/ufo-sightings-around-the-world).

    Args:
        local_save: Whether to save the data locally if it doesn't exists.
        read_kwargs: kwargs to pass to pd.read_csv

    Returns:
        Sighting level data for UFOs.

    """
    name = "ufo_sighting_data"
    read_kwargs = {"low_memory": False, **read_kwargs}
    save_kwargs = {"index": False}

    df = _load_data(
        name, read_kwargs=read_kwargs, save_kwargs=save_kwargs, local_save=local_save
    )
    df["Date_time"] = pd.to_datetime(
        df["Date_time"]
        .str.replace(" 24:00", " 23:59")
        .pipe(pd.to_datetime, format="mixed", dayfirst=True)
    )

    return df
