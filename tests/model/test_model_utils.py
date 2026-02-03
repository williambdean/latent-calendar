import pytest

import warnings

import pandas as pd
import numpy as np

from latent_calendar.model.latent_calendar import LatentCalendar
from latent_calendar.model.utils import transform_on_dataframe, predict_on_dataframe


@pytest.fixture
def X() -> np.ndarray:
    ncols = 15
    nrows = 25
    return np.ones((nrows, ncols))


@pytest.fixture
def df(X) -> pd.DataFrame:
    data = pd.DataFrame(X)
    data.index.name = "index"
    data.columns.name = "columns"

    return data


@pytest.fixture
def n_components() -> int:
    return 10


@pytest.fixture
def model_with_X(X, n_components) -> LatentCalendar:
    model = LatentCalendar(random_state=42, n_components=n_components)

    return model.fit(X)


@pytest.fixture
def model_with_df(df, n_components) -> LatentCalendar:
    model = LatentCalendar(random_state=42, n_components=n_components)
    return model.fit(df)


@pytest.mark.parametrize("model_name", ["model_with_X", "model_with_df"])
@pytest.mark.parametrize("func", [transform_on_dataframe, predict_on_dataframe])
def test_dataframe_with_model_with_X(
    request, df, n_components, model_name, func
) -> None:
    nrows = 25
    ncols = 15

    model = request.getfixturevalue(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = func(df=df, model=model)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == nrows
    assert result.index.name == "index"

    if func.__name__ == "predict_on_dataframe":
        assert result.columns.name == "columns"
        assert len(result.columns) == ncols
    else:
        assert result.columns.name is None
        assert len(result.columns) == n_components
