import pytest

import pandas as pd
import numpy as np

from latent_calendar.plot.iterate import (
    iterate_matrix,
    iterate_long_array,
    iterate_dataframe,
    IterConfig,
    VocabIterConfig,
    CalendarData,
)


@pytest.mark.parametrize("nrows, ncols", [(7, 5), (7, 2), (7, 24)])
def test_iterate_matrix(nrows: int, ncols: int) -> None:
    data = np.ones((nrows, ncols))

    iterator = iterate_matrix(data)

    for i, value in enumerate(iterator):
        assert isinstance(value, CalendarData)

    assert i + 1 == nrows * ncols


@pytest.mark.parametrize(
    "data",
    [
        # Not enough rows
        np.ones((6, 5)),
        # Not right amount of dims
        np.ones((7, 5, 2)),
        np.ones((5,)),
    ],
)
def test_iterator_matrix_raise_value_error(data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        next(iterate_matrix(data))


def test_iterator_long_array() -> None:
    size = 7 * 24
    data = np.ones((size,))

    iterator = iterate_long_array(data)

    for i, value in enumerate(iterator):
        assert isinstance(value, CalendarData)

    assert i + 1 == size


@pytest.mark.parametrize(
    "data",
    [
        # Without value
        {"day": [1, 2, 3], "start": [1, 2, 3], "end": [1, 2, 3]},
        # With value
        {"day": [1, 2], "start": [1, 2], "end": [1, 2], "value": [1, 2]},
    ],
)
def test_iter_config_and_dataframe(data: dict[str, list[int]]) -> None:
    config = IterConfig(day="day", start="start", end="end")

    df = pd.DataFrame(data)

    extracted_cols = config.extract_columns(df)

    assert isinstance(extracted_cols, tuple)

    iterator = iterate_dataframe(df, config)

    for i, values in enumerate(iterator):
        assert isinstance(values, CalendarData)

    assert i + 1 == len(df)


def test_iter_config_missing_column() -> None:
    data = {
        "day": [1, 2, 3],
        "start": [1, 2, 3],
        "end": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    config = IterConfig("day", "start", "not_end", None)

    with pytest.raises(KeyError):
        next(iterate_dataframe(df, config))


@pytest.mark.parametrize(
    "data",
    [
        # Without value
        {"vocab": ["01 01", "01 02", "01 03"]},
        # With value
        {"vocab": ["01 01", "01 02", "03 01"], "value": [1, 2, 3]},
    ],
)
def test_iter_vocab_config_and_dataframe(data: dict[str, list[int]]) -> None:
    config = VocabIterConfig(vocab="vocab", value="value")

    df = pd.DataFrame(data)

    extracted_cols = config.extract_columns(df)

    assert isinstance(extracted_cols, tuple)

    iterator = iterate_dataframe(df, config)

    for i, values in enumerate(iterator):
        assert isinstance(values, CalendarData)

    assert i + 1 == len(df)
