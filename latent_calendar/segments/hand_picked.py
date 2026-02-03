"""Create hand picked segments on the calendar.

Examples:
    Create some segments for a calendar:

    ```python
    mornings = create_box_segment(
        day_start=0, day_end=7, hour_start=6, hour_end=11, name="Mornings"
    )
    afternoons = create_box_segment(
        day_start=0, day_end=7, hour_start=11, hour_end=16, name="Afternoons"
    )
    evenings = create_box_segment(
        day_start=0, day_end=7, hour_start=16, hour_end=21, name="Evenings"
    )

    df_segments = stack_segments([
        mornings,
        afternoons,
        evenings,
    ])

    df_segments.cal.plot_by_row()
    ```

    ![New Segments](./../images/new-segments.png)

"""

import itertools

import pandas as pd
import numpy as np

from latent_calendar.plot.elements import create_default_days
from latent_calendar.const import (
    HOURS_IN_DAY,
    DAYS_IN_WEEK,
    FULL_VOCAB,
    format_dow_hour,
)
from latent_calendar.vocab import DOWHour


def create_empty_template() -> pd.DataFrame:
    """Create blank template in order"""
    index = pd.Index(range(HOURS_IN_DAY), name="hour_start")
    return pd.DataFrame(
        np.nan,
        index=index,
        columns=create_default_days(),
    )


def create_blank_segment_series() -> pd.Series:
    """Helper for making segments programatically."""
    return pd.Series(0, index=FULL_VOCAB)


def create_series_for_range(start: DOWHour, end: DOWHour) -> pd.Series:
    """Create a series for a range of hours with ones for those in range."""
    ser = create_blank_segment_series()

    if start.is_after(end):
        end, start = start, end
        negate = True
    else:
        negate = False

    if isinstance(ser.index, pd.MultiIndex):
        start_idx = pd.IndexSlice[start.dow, start.hour]
        end_idx = pd.IndexSlice[end.dow, end.hour - 1]
    else:
        start_idx = format_dow_hour(start.dow, start.hour)
        end_idx = format_dow_hour(end.dow, end.hour - 1)

    ser.loc[start_idx:end_idx] = 1

    if negate:
        ser = (ser - 1) * -1

    return ser.astype(int)


def get_vocab_for_range(start: DOWHour, end: DOWHour) -> list[str]:
    """Get the vocab for a range of hours."""
    return (
        create_series_for_range(start=start, end=end)
        .loc[lambda x: x == 1]
        .index.tolist()
    )


def create_hourly_segment(start: DOWHour, end: DOWHour, name: str) -> pd.Series:
    """Highlight from start until end."""
    return create_series_for_range(start=start, end=end).rename(name)


def create_box_segment(
    day_start: int,
    day_end: int,
    hour_start: int,
    hour_end: int,
    name: str | None = None,
) -> pd.Series:
    """Programmatically make segment of box described by inputs."""
    ser = create_blank_segment_series()

    for dow in range(day_start, day_end):
        start = DOWHour(dow=dow, hour=hour_start)
        end = DOWHour(dow=dow, hour=hour_end)

        ser += create_series_for_range(start=start, end=end)

    name = name or f"{day_start}-{day_end} {hour_start}-{hour_end}"
    return ser.rename(name)


SEGMENT = pd.Series | pd.DataFrame


def stack_segments(segments: list[SEGMENT]) -> pd.DataFrame:
    """Stack segments into a single dataframe."""
    segments = [seg.T if isinstance(seg, pd.DataFrame) else seg for seg in segments]
    return pd.concat(segments, axis=1).T


def create_dow_segments() -> pd.DataFrame:
    """Programmatically make the DOW segments.

    Each row is just each day of the week.

    Returns:
        DataFrame in the df_segments wide format

    """
    segments = []

    for i, day in enumerate(create_default_days()):
        day_number = str(i).zfill(2)
        name = f"{day_number}-{day}"

        start = DOWHour(dow=i, hour=0)
        end = DOWHour(dow=i, hour=24)

        segments.append(create_hourly_segment(start=start, end=end, name=name))

    return stack_segments(segments)


def create_every_hour_segments() -> pd.DataFrame:
    """Programmatically segments for every hour

    Each row is just each time slot

    Returns:
        DataFrame in the df_segments wide format

    """
    segments = []

    for dow, hour in itertools.product(range(DAYS_IN_WEEK), range(HOURS_IN_DAY)):
        name = format_dow_hour(dow, hour)

        start = DOWHour(dow=dow, hour=hour)
        end = DOWHour(dow=dow, hour=hour + 1)
        segments.append(create_hourly_segment(start=start, end=end, name=name))

    return stack_segments(segments)
