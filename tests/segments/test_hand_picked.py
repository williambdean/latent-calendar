import pytest

import pandas as pd
import numpy as np

from latent_calendar.const import TIME_SLOTS, DAYS_IN_WEEK, FULL_VOCAB
from latent_calendar.segments.hand_picked import (
    create_blank_segment_series,
    get_vocab_for_range,
    create_empty_template,
    create_hourly_segment,
    create_dow_segments,
    create_every_hour_segments,
    stack_segments,
)
from latent_calendar.vocab import DOWHour


def test_create_empty_template() -> None:
    df = create_empty_template()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (24, 7)


@pytest.mark.parametrize(
    "start, end, answer",
    [
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=1), [(0, 0)]),
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=2), [(0, 0), (0, 1)]),
    ],
)
def test_get_vocab_for_range(start, end, answer) -> None:
    vocab = get_vocab_for_range(start=start, end=end)

    assert isinstance(vocab, list)
    assert vocab == answer


def df_strat_tests(df_strat: pd.DataFrame, index) -> None:
    assert isinstance(df_strat, pd.DataFrame)
    pd.testing.assert_index_equal(df_strat.index, pd.Index(index).sort_values())


@pytest.mark.parametrize(
    "start, end, answer",
    [
        # Single Day
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=1), 1),
        (DOWHour(dow=0, hour=0), DOWHour(dow=0, hour=24), 24),
        # Multi Day
        (DOWHour(dow=0, hour=23), DOWHour(dow=1, hour=1), 2),
        # Over Sunday
        (DOWHour(dow=6, hour=23), DOWHour(dow=0, hour=1), 2),
    ],
)
def test_create_hour_strat(start, end, answer) -> None:
    ser = create_hourly_segment(start=start, end=end, name="Test Name")

    assert isinstance(ser, pd.Series)
    assert np.issubdtype(ser.dtype, np.integer)
    assert ser.sum() == answer
    assert ser.name == "Test Name"


def test_create_dow_strategy() -> None:
    df_strat = create_dow_segments()

    assert isinstance(df_strat, pd.DataFrame)
    assert df_strat.shape == (DAYS_IN_WEEK, TIME_SLOTS)


def test_create_every_hour_strategy() -> None:
    df_strat = create_every_hour_segments()

    assert isinstance(df_strat, pd.DataFrame)
    assert df_strat.shape == (TIME_SLOTS, TIME_SLOTS)


@pytest.fixture
def first() -> pd.Series:
    return create_blank_segment_series().rename("first")


@pytest.fixture
def second() -> pd.Series:
    return create_blank_segment_series().rename("second")


@pytest.fixture
def third() -> pd.Series:
    return create_blank_segment_series().rename("third")


@pytest.fixture
def group(first, second) -> pd.DataFrame:
    return stack_segments([first, second])


def test_stack_segments_all_series(group: pd.DataFrame) -> None:
    pd.testing.assert_index_equal(group.index, pd.Index(["first", "second"]))
    pd.testing.assert_index_equal(group.columns, FULL_VOCAB)


def test_stack_segments_all_df(group: pd.DataFrame) -> None:
    df_stack = stack_segments([group, group])

    pd.testing.assert_index_equal(
        df_stack.index, pd.Index(["first", "second", "first", "second"])
    )
    pd.testing.assert_index_equal(group.columns, FULL_VOCAB)


def test_stack_segments_mixed(group: pd.DataFrame, third: pd.Series) -> None:
    df_stack = stack_segments([group, third])

    pd.testing.assert_index_equal(
        df_stack.index, pd.Index(["first", "second", "third"])
    )
    pd.testing.assert_index_equal(group.columns, FULL_VOCAB)
