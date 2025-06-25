import pytest

import narwhals as nw
import pandas as pd
import polars as pl
import polars.testing

from latent_calendar.transformers import (
    prop_into_day,
    CalendarTimestampFeatures,
    HourDiscretizer,
    VocabTransformer,
    create_timestamp_feature_pipeline,
    create_raw_to_vocab_transformer,
)


@pytest.fixture
def sample_timestamp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "datetime": pd.to_datetime(
                [
                    "2021-01-01 12:00",
                    "2021-01-01 13:55",
                    "2021-01-01 14:05",
                    "2021-01-01 14:55",
                ]
            ),
            "another_grouping": [1, 1, 1, 2],
        },
        index=pd.Index(["first", "second", "third", "fourth"]),
    )


@pytest.fixture
def sample_hour_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": [0, 1.5, 2, 3.25, 4.35],
        }
    )


@pytest.mark.parametrize(
    "date", ["2023-07-01", "2023-01-01", "2020-01-01", "1970-01-01"]
)
def test_prop_into_day_series(date) -> None:
    times = ["00:00", "01:00", "12:00", "23:59"]
    answers = [0.0, 1 / 24, 0.5, 0.9993]
    dates = pd.Series(pd.to_datetime([f"{date} {time}" for time in times]))

    @nw.narwhalify
    def df_prop_into_day(df):
        col = nw.col("datetime")
        return df.with_columns(new_col=prop_into_day(col.dt))

    results = (
        pd.DataFrame({"datetime": dates}).pipe(df_prop_into_day)["new_col"].rename(None)
    )
    answer = pd.Series(answers)

    pd.testing.assert_series_equal(results, answer, atol=0.001)


@pytest.mark.parametrize("pandas_output", [True, False])
def test_calendar_timestamp_features(
    sample_timestamp_df: pd.DataFrame,
    pandas_output: bool,
) -> None:
    timestamp_features = CalendarTimestampFeatures(
        timestamp_col="datetime",
    )
    if pandas_output:
        timestamp_features.set_output(transform="pandas")

    df_result = timestamp_features.fit_transform(sample_timestamp_df)

    assert isinstance(df_result, pd.DataFrame)
    cols_to_check = ["hour", "day_of_week"]
    for col in cols_to_check:
        assert col in df_result.columns
        assert col not in sample_timestamp_df.columns

    assert len(df_result.columns) == len(sample_timestamp_df.columns) + len(
        cols_to_check
    )


@pytest.mark.parametrize(
    "minutes, answer",
    [
        (60, [0, 1, 2, 3, 4]),
        (30, [0, 1.5, 2, 3, 4]),
        (15, [0, 1.5, 2, 3.25, 4.25]),
        (120, [0, 0, 2, 2, 4]),
        (180, [0, 0, 0, 3, 3]),
    ],
)
def test_hour_discretizer(sample_hour_df: pd.DataFrame, minutes, answer) -> None:
    col = "hour"
    transformer = HourDiscretizer(col=col, minutes=minutes)

    df_result = transformer.fit_transform(sample_hour_df)
    pd.testing.assert_series_equal(
        df_result[col], pd.Series(answer).rename(col), atol=0.001
    )


def test_timestamp_features(sample_timestamp_df: pd.DataFrame) -> None:
    pipe = create_timestamp_feature_pipeline(timestamp_col="datetime")

    df_result = pipe.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert len(df_result) == len(sample_timestamp_df)


def test_raw_to_vocab(sample_timestamp_df) -> None:
    pipeline = create_raw_to_vocab_transformer(id_col="id", timestamp_col="datetime")

    df_result = pipeline.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(df_result.index, pd.Index)
    assert len(df_result) == len(sample_timestamp_df["id"].unique())


def test_raw_to_vocab_with_groups(sample_timestamp_df) -> None:
    pipeline = create_raw_to_vocab_transformer(
        id_col="id", timestamp_col="datetime", additional_groups=["another_grouping"]
    )

    df_result = pipeline.fit_transform(sample_timestamp_df.copy())

    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(df_result.index, pd.MultiIndex)


@pytest.fixture
def polars_sample_timestamp_df(sample_timestamp_df) -> pl.DataFrame:
    return pl.from_pandas(sample_timestamp_df)


@pytest.mark.parametrize("as_multiindex", [True, False])
def test_polars_run_through(polars_sample_timestamp_df, as_multiindex: bool) -> None:
    transformer = create_raw_to_vocab_transformer(
        id_col="id",
        timestamp_col="datetime",
        as_multiindex=as_multiindex,
        widen=False,
    )

    # Expected output:
    # ┌─────┬─────────────┬──────┬────────────┐
    # │ id  ┆ day_of_week ┆ hour ┆ num_events │
    # │ --- ┆ ---         ┆ ---  ┆ ---        │
    # │ i64 ┆ i8          ┆ i64  ┆ i32        │
    # ╞═════╪═════════════╪══════╪════════════╡
    # │ 1   ┆ 4           ┆ 12   ┆ 1          │
    # │ 2   ┆ 4           ┆ 14   ┆ 2          │
    # │ 1   ┆ 4           ┆ 13   ┆ 1          │

    if not as_multiindex:
        data = {
            "vocab": ["04 12", "04 14", "04 13"],
        }
    else:
        data = {
            "day_of_week": [4, 4, 4],
            "hour": [12, 14, 13],
        }

    print(f"{as_multiindex = }")
    df_result = transformer.fit_transform(polars_sample_timestamp_df)
    print(df_result)
    polars.testing.assert_frame_equal(
        df_result,
        pl.DataFrame(
            {
                "id": [1, 2, 1],
                **data,
                "num_events": [1, 2, 1],
            }
        ),
        check_row_order=False,
        check_dtypes=False,
    )


@pytest.fixture
def polars_input() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "day_of_week": [1, 2, 3],
            "hour": [0, 1, 2],
        }
    )


def test_polars_vocab_transformer(polars_input) -> None:
    transformer = VocabTransformer()
    df_result = transformer.fit_transform(polars_input)

    polars.testing.assert_frame_equal(
        df_result,
        pl.DataFrame(
            {
                "day_of_week": [1, 2, 3],
                "hour": [0, 1, 2],
                "vocab": ["01 00", "02 01", "03 02"],
            }
        ),
    )
