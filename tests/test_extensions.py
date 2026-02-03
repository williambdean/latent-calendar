import pytest

import polars as pl
import polars.testing
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import latent_calendar  # noqa
from latent_calendar.const import (
    TIME_SLOTS,
    FULL_VOCAB,
    DAYS_IN_WEEK,
    HOURS_IN_DAY,
    create_full_vocab,
)


@pytest.fixture
def ser() -> pd.Series:
    return pd.Series(pd.date_range("2023-01-01", "2023-01-14", freq="h"))


def test_series_extensions(ser) -> None:
    assert hasattr(ser, "cal")

    ax = ser.cal.plot()
    assert isinstance(ax, plt.Axes)


@pytest.fixture
def ser_row(ser) -> pd.Series:
    return pd.Series(1, index=FULL_VOCAB)


@pytest.mark.parametrize(
    "level, axis",
    [
        ("day_of_week", 1),
        (0, 1),
        ("hour", 0),
        (1, 0),
    ],
)
def test_series_conditional_probabilities(ser_row, level, axis) -> None:
    result = ser_row.cal.conditional_probabilities(level=level).unstack().sum(axis=axis)
    # All the probabilities should sum to 1

    assert (result.round() == 1).all()


@pytest.fixture
def df() -> pd.DataFrame:
    """Generate some fake data."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    ).T


@pytest.fixture
def df_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment 1": [1, 0, 1],
            "segment 2": [0, 1, 0],
            "segment 3": [1, 1, 0],
        }
    ).T


def test_divide_by_max(df) -> None:
    df_max_result = pd.DataFrame(
        {
            "a": [1 / 3, 2 / 3, 1],
            "b": [4 / 6, 5 / 6, 1],
        }
    ).T

    pd.testing.assert_frame_equal(
        df.cal.divide_by_max(),
        df_max_result,
    )


def test_divide_by_sum(df) -> None:
    df_probs_result = pd.DataFrame(
        {
            "a": [1 / 6, 2 / 6, 3 / 6],
            "b": [4 / 15, 5 / 15, 6 / 15],
        }
    ).T

    pd.testing.assert_frame_equal(
        df.cal.divide_by_sum(),
        df_probs_result,
    )


def test_divide_by_even_rate(df) -> None:
    df_even_rate_result = pd.DataFrame(
        {
            "a": [1 * 3, 2 * 3, 3 * 3],
            "b": [4 * 3, 5 * 3, 6 * 3],
        }
    ).T

    pd.testing.assert_frame_equal(
        df.cal.divide_by_even_rate(),
        df_even_rate_result,
    )


def test_even_rate_probability_distribution(df) -> None:
    df_even_rate_result = pd.DataFrame(
        {
            "a": [0.5, 1.0, 1.5],
            "b": [0.8, 1.0, 1.2],
        }
    ).T

    df_probs = df.div(df.sum(axis=1), axis=0)

    pd.testing.assert_frame_equal(
        df_probs.cal.divide_by_even_rate(),
        df_even_rate_result,
    )


def test_all_dataframe_extensions(df, df_segments) -> None:
    assert hasattr(df, "cal")

    pd.testing.assert_frame_equal(
        df.cal.sum_over_segments(df_segments),
        pd.DataFrame(
            {
                "segment 1": [4, 10],
                "segment 2": [2, 5],
                "segment 3": [3, 9],
            },
            index=df.index,
        ),
    )


@pytest.fixture
def df_long(ser) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": rng.choice(["a", "b", "c"], size=len(ser)),
            "another_group": rng.choice(["x"], size=len(ser)),
            "timestamp": ser,
        }
    )


def test_long_dataframe_extensions(df_long) -> None:
    assert hasattr(df_long, "cal")

    df_wide = df_long.cal.aggregate_events("group", "timestamp")
    assert df_wide.shape == (3, TIME_SLOTS)

    df_wide = df_long.cal.aggregate_events(["group", "another_group"], "timestamp")
    assert df_wide.shape == (3, TIME_SLOTS)

    ax = df_long.cal.plot("timestamp")
    assert isinstance(ax, plt.Axes)

    df_long.cal.plot_across_column("timestamp", "group")


@pytest.fixture
def df_agg(df_long) -> pd.DataFrame:
    return (
        df_long.cal.timestamp_features("timestamp")
        .groupby(["group", "day_of_week", "hour"])
        .size()
        .rename("num_events")
        .to_frame()
    )


def test_agg_dataframe_extensions(df_agg) -> None:
    assert hasattr(df_agg, "cal")

    df_wide = df_agg.cal.widen("num_events")
    assert df_wide.shape == (3, TIME_SLOTS)

    with pytest.raises(ValueError):
        df_agg.reset_index(0).cal.widen("num_events")

    df_false_order = df_agg.swaplevel(1, 0).cal.widen("num_events")
    assert isinstance(df_false_order, pd.DataFrame)
    assert df_false_order.sum().sum() == 0


@pytest.fixture
def df_wide() -> pd.DataFrame:
    nrows = 25
    data = np.ones((nrows, TIME_SLOTS))
    return pd.DataFrame(data, columns=FULL_VOCAB)


class MockModel:
    def __init__(self) -> None:
        self.n_components = 3

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), self.n_components))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), TIME_SLOTS))


def test_wide_dataframe_extensions(df_wide: pd.DataFrame) -> None:
    assert hasattr(df_wide, "cal")
    nrows = len(df_wide)
    df_dow = df_wide.cal.sum_over_vocab("dow")

    df_dow_answer = pd.DataFrame(
        np.ones((nrows, DAYS_IN_WEEK)) * HOURS_IN_DAY,
        index=df_wide.index,
        columns=pd.Index(range(DAYS_IN_WEEK), name="day_of_week"),
    )

    pd.testing.assert_frame_equal(df_dow, df_dow_answer)

    df_hour = df_wide.cal.sum_over_vocab("hour")

    df_hour_answer = pd.DataFrame(
        np.ones((nrows, HOURS_IN_DAY)) * DAYS_IN_WEEK,
        index=df_wide.index,
        columns=pd.Index(range(HOURS_IN_DAY), name="hour"),
    )

    pd.testing.assert_frame_equal(df_hour, df_hour_answer)

    df_wide.head(1).cal.plot_by_row()

    mock_model = MockModel()
    assert df_wide.cal.transform(model=mock_model).shape == (
        nrows,
        mock_model.n_components,
    )
    assert df_wide.cal.predict(model=mock_model).shape == (nrows, TIME_SLOTS)

    for head_rows in [1, 3]:
        df_plot = df_wide.head(head_rows).copy()

        df_plot.cal.plot_profile_by_row(model=mock_model, include_components=False)
        df_plot.cal.plot_profile_by_row(model=mock_model, include_components=True)
        df_plot.cal.plot_raw_and_predicted_by_row(model=mock_model)
        df_plot.cal.plot_model_predictions_by_row(df_plot, model=mock_model)

    for next_hours in [1, 2, 3]:
        df_answer = pd.DataFrame(
            np.ones((nrows, TIME_SLOTS)) * (1 + next_hours),
            index=df_wide.index,
            columns=FULL_VOCAB,
        )

        pd.testing.assert_frame_equal(
            df_wide.cal.sum_next_hours(hours=next_hours), df_answer
        )


@pytest.fixture
def df_wide_subset() -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(
        [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ],
        names=["day_of_week", "hour"],
    )

    data = np.ones((3, 6))
    return pd.DataFrame(data, columns=columns).sort_index(axis=1)


@pytest.mark.parametrize(
    "level, answer",
    [
        ("day_of_week", 1 / 3),
        (0, 1 / 3),
        ("hour", 1 / 2),
        (1, 1 / 2),
    ],
)
def test_dataframe_conditional_probabilities(
    df_wide_subset: pd.DataFrame, level, answer
) -> None:
    result = df_wide_subset.cal.conditional_probabilities(level=level)
    expected = pd.DataFrame(
        answer, index=df_wide_subset.index, columns=df_wide_subset.columns
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.fixture(scope="module")
def timestamps() -> list[str]:
    return [
        # 2021-01-01 is a Friday
        "2021-01-01 00:00:00",
        "2021-01-01 00:30:00",
        "2021-01-01 01:00:00",
        "2021-01-01 01:30:00",
        "2021-01-01 01:45:00",
        "2021-01-01 01:50:00",
    ]


def test_series_aggregate_events(timestamps) -> None:
    name = "events"
    ser = pd.Series(
        pd.to_datetime(timestamps),
        name=name,
    )

    result = ser.cal.aggregate_events(minutes=30)

    expected = pd.Series(
        0,
        index=create_full_vocab(days_in_week=7, minutes=30),
        name=name,
        dtype=result.dtype,
    )
    expected.loc[(4, 0)] = 1
    expected.loc[(4, 0.5)] = 1
    expected.loc[(4, 1)] = 1
    expected.loc[(4, 1.5)] = 3

    pd.testing.assert_series_equal(result, expected)


@pytest.fixture
def df_polars(timestamps) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": range(len(timestamps)),
            "datetime": pl.Series(timestamps).str.to_datetime(),
        }
    )


def test_polars_dataframe_extensions(df_polars) -> None:
    assert hasattr(df_polars, "cal")

    cal = df_polars.cal

    df_features = cal.timestamp_features("datetime")

    assert df_features.shape == (6, 5)

    df_agg = cal.aggregate_events("id", "datetime")
    assert df_agg.shape == (6, 4)
