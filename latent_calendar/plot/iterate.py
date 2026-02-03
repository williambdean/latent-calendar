"""Generalize the iteration to support different data formats. Namely,

- 2d numpy array
- 1d numpy array (long format)
- pandas Series
- pandas DataFrame with various columns

This powers the calendar plot and is passed into the `plot_calendar` function.

Examples:
    Plot calendar based on 1d numpy array.

    ```python
    import numpy as np

    from latent_calendar.plot import plot_calendar
    from latent_calendar.plot.iterate import iterate_long_array

    data = np.ones(7 * 24)
    plot_calendar(
        iterate_long_array(data),
    )
    ```

    Plot calendar based on 2d numpy array.

    ```python
    from latent_calendar.plot import plot_calendar

    data = np.ones((7, 24))
    plot_calendar(
        iterate_matrix(data),
    )
    ```

    Plot calendar for every half hour instead of every hour. **NOTE:** This happens automatically!

    ```python
    from latent_calendar.plot import plot_calendar

    data = np.ones((7, 24 * 2))
    plot_calendar(
        iterate_matrix(data),
    )
    ```

"""

from dataclasses import dataclass
from itertools import repeat
from typing import Any, Generator, Iterable

import numpy as np
import pandas as pd

from latent_calendar.const import DAYS_IN_WEEK, HOURS_IN_DAY
from latent_calendar.transformers import prop_into_day


@dataclass
class CalendarData:
    """All the data that goes into calendar plot."""

    day: int
    start: float
    end: float
    value: float


CALENDAR_ITERATION = Generator[CalendarData, None, None]


def iterate_matrix(calendar_data: np.ndarray) -> CALENDAR_ITERATION:
    """Iterates the calendar matrix of values."""
    if calendar_data.ndim != 2:
        raise ValueError(f"Data must be 2d not of shape {calendar_data.shape}")

    n_days, n_hours = calendar_data.shape

    if n_days != DAYS_IN_WEEK:
        raise ValueError(f"Data must have {DAYS_IN_WEEK} days not {n_days}")

    step_size = HOURS_IN_DAY / n_hours

    for day, hours in enumerate(calendar_data):
        for hour, value in enumerate(hours):
            start = hour * step_size
            end = start + step_size
            yield CalendarData(day, start, end, value)


def iterate_long_array(calendar_data: np.ndarray) -> CALENDAR_ITERATION:
    matrix = calendar_data.reshape(DAYS_IN_WEEK, -1)
    yield from iterate_matrix(matrix)


def iterate_series(calendar_data: pd.Series) -> CALENDAR_ITERATION:
    long_array = calendar_data.to_numpy()
    yield from iterate_long_array(long_array)


REPEATABLE = pd.Series | Iterable[float]


VALUE_DEFAULT = 1

FRAME_ITER = tuple[pd.Series, pd.Series, pd.Series, REPEATABLE]


class DataFrameConfig:
    @property
    def columns(self) -> list[str]:
        raise NotImplementedError("columns property needs to be implemented.")

    def extract_columns(self) -> FRAME_ITER:
        raise NotImplementedError("extract_columns method needs to be implemented.")

    def _check_columns(self, df: pd.DataFrame) -> None:
        try:
            df.loc[:, self.columns]
        except KeyError as e:
            msg = f"Missing column in the DataFrame. Please alter either DataFrame columns or IterConfig. {e}"
            raise KeyError(msg)

    def _default_repeat(
        self,
        df: pd.DataFrame,
        key: str | None,
        default_value: Any,
    ) -> REPEATABLE:
        return repeat(default_value) if key not in df.columns else df[key]


@dataclass
class IterConfig(DataFrameConfig):
    """Small wrapper to hold the column mapping in DataFrame."""

    day: str = "day_of_week"
    start: str = "hour_start"
    end: str = "hour_end"
    value: str = "value"

    @property
    def columns(self) -> list[str]:
        return [self.day, self.start, self.end]

    def extract_columns(self, df: pd.DataFrame) -> FRAME_ITER:
        self._check_columns(df)
        return (
            df[self.day],
            df[self.start],
            df[self.end],
            self._default_repeat(df, self.value, VALUE_DEFAULT),
        )


@dataclass
class StartEndConfig(DataFrameConfig):
    start: str
    end: str | None = None
    minutes: int | None = None
    value: str = "value"

    def __post_init__(self) -> None:
        if self.end is not None and self.minutes is not None:
            raise ValueError("Only one of end or minutes can be specified.")

        if self.end is None and self.minutes is None:
            self.minutes = 5

    @property
    def columns(self) -> list[str]:
        return [self.start, self.end]

    def extract_columns(self, df: pd.DataFrame) -> FRAME_ITER:
        dow = df[self.start].dt.day_of_week

        start = prop_into_day(df[self.start].dt) * HOURS_IN_DAY

        if self.end is None:
            end = start + (self.minutes / 60)
        else:
            end = prop_into_day(df[self.end].dt) * HOURS_IN_DAY

            before = end < start
            end[before] += HOURS_IN_DAY

        return (
            dow,
            start,
            end,
            self._default_repeat(df, self.value, VALUE_DEFAULT),
        )


@dataclass
class VocabIterConfig(DataFrameConfig):
    """Small wrapper to hold the column mapping in the DataFrame."""

    vocab: str = "vocab"
    value: str = "value"

    @property
    def columns(self) -> list[str]:
        return [self.vocab]

    def extract_columns(self, df: pd.DataFrame) -> FRAME_ITER:
        self._check_columns(df)

        day = df[self.vocab].str.split(" ").apply(lambda x: int(x[0]))
        start = df[self.vocab].str.split(" ").apply(lambda x: int(x[1]))

        return (
            day,
            start,
            start + 1,
            self._default_repeat(df, self.value, VALUE_DEFAULT),
        )


def iterate_dataframe(
    df: pd.DataFrame,
    config: DataFrameConfig,
) -> CALENDAR_ITERATION:
    """Iterate the calendar data in DataFrame form based on config.

    Args:
        df: DataFrame with calendar data.
        config: Configuration to describe what columns to use.

    """
    for values in zip(*config.extract_columns(df)):
        yield CalendarData(*values)
