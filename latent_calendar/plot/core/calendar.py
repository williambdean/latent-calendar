from itertools import repeat
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from latent_calendar.plot.colors import (
    CMAP,
    create_default_cmap,
    create_default_divergent_cmap,
    ColorMap,
)
from latent_calendar.plot.elements import (
    CalendarEvent,
    update_display_settings,
    update_start,
    configure_axis,
    DisplaySettings,
    GridLines,
    TimeLabeler,
    DayLabeler,
)
from latent_calendar.plot.grid_settings import default_axes_and_grid_axes
from latent_calendar.plot.iterate import (
    CALENDAR_ITERATION,
    DataFrameConfig,
    iterate_dataframe,
    iterate_series,
    iterate_long_array,
)


def plot_blank_calendar(
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
    display_settings: DisplaySettings | None = None,
    ax: plt.Axes | None = None,
    grid_lines: GridLines = GridLines(),
    monday_start: bool = True,
) -> plt.Axes:
    """Create a blank calendar with no data

    Args:
        day_labeler: instance in order to configure the day labels
        time_labeler: instance in order to configure the time labels
        display_settings: override of the display settings in the calendar
        ax: Optional axes to plot on
        grid_lines: GridLines instance
        monday_start: whether to start the week on Monday or Sunday

    Returns:
        Modified matplotlib axis

    """
    update_start(day_labeler=day_labeler, monday_start=monday_start)

    if display_settings is not None:
        update_display_settings(
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            display_settings=display_settings,
        )

    ax = ax if ax is not None else plt.gca()

    configure_axis(ax=ax, day_labeler=day_labeler, time_labeler=time_labeler)
    grid_lines.configure_grid(ax=ax)

    return ax


def plot_calendar(
    calendar_iter: CALENDAR_ITERATION,
    *,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
    display_settings: DisplaySettings | None = None,
    cmap: CMAP | None = None,
    alpha: float | None = None,
    ax: plt.Axes | None = None,
    grid_lines: GridLines = GridLines(),
    monday_start: bool = True,
) -> plt.Axes:
    """Plot a calendar from generator of values.

    This can plot both numpy matrix and DataFrame values as long as the iterable fits CALENDAR_ITERATION definition

    Args:
        calendar_iter: CALENDAR_ITERATION
        day_labeler: instance in order to configure the day labels
        time_labeler: instance in order to configure the time labels
        display_settings: override of the display settings in the calendar
        cmap: function that maps floats to string colors
        ax: Optional axes to plot on
        grid_lines: GridLines instance
        monday_start: whether to start the week on Monday or Sunday

    Returns:
        Modified matplotlib axis

    """
    ax = plot_blank_calendar(
        day_labeler=day_labeler,
        time_labeler=time_labeler,
        display_settings=display_settings,
        ax=ax,
        grid_lines=grid_lines,
        monday_start=monday_start,
    )

    if cmap is None:

        def cmap(x: float) -> str:
            return "lightblue"

    for calendar_data in calendar_iter:
        event = CalendarEvent.from_calendar_data(calendar_data=calendar_data)

        event.plot(
            ax=ax,
            facecolor=cmap(calendar_data.value),
            alpha=alpha,
            monday_start=monday_start,
        )

    return ax


def plot_series_as_calendar(
    series: pd.Series,
    *,
    grid_lines: GridLines = GridLines(),
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
    cmap: CMAP | None = None,
    alpha: float | None = None,
    ax: plt.Axes | None = None,
    monday_start: bool = True,
) -> plt.Axes:
    """Simple Wrapper about plot_calendar in order to plot Series in various formats

    Args:
        series: Series in format with index as datetime and values as float
        grid_lines: GridLines instance
        day_labeler: instance in order to configure the day labels
        time_labeler: instance in order to configure the time labels
        cmap: function that maps floats to string colors
        alpha: alpha level of each rectangle
        ax: optional axis to plot on
        monday_start: whether to start the week on Monday or Sunday

    Returns:
        new or modified axes

    """
    if cmap is None:
        cmap = create_default_cmap(value=series.to_numpy().max())

    return plot_calendar(
        iterate_series(series),
        day_labeler=day_labeler,
        time_labeler=time_labeler,
        cmap=cmap,
        alpha=alpha,
        ax=ax,
        monday_start=monday_start,
        grid_lines=grid_lines,
    )


def plot_dataframe_as_calendar(
    df: pd.DataFrame,
    config: DataFrameConfig,
    *,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
    grid_lines: GridLines = GridLines(),
    cmap: CMAP | None = None,
    alpha: float | None = None,
    ax: plt.Axes | None = None,
    monday_start: bool = True,
) -> plt.Axes:
    """Simple Wrapper about plot_calendar in order to plot DataFrame in various formats

    Args:
        df: DataFrame in format with columns in config instance
        config: DataFrameConfig
        day_labeler: instance in order to configure the day labels
        time_labeler: instance in order to configure the time labels
        grid_lines: GridLines instance
        cmap: function that maps floats to string colors
        alpha: alpha level of each rectangle
        ax: optional axis to plot on
        monday_start: whether to start the week on Monday or Sunday

    Returns:
        new or modified axes

    """
    return plot_calendar(
        iterate_dataframe(df, config),
        day_labeler=day_labeler,
        time_labeler=time_labeler,
        cmap=cmap,
        alpha=alpha,
        ax=ax,
        monday_start=monday_start,
        grid_lines=grid_lines,
    )


TITLE_FUNC = Callable[[Any, pd.Series], str]


def default_title_func(idx: Any, row: pd.Series) -> str:
    return idx


CMAP_GENERATOR = Generator[CMAP, None, None]


def create_alternating_cmap(max_values: pd.Series) -> CMAP_GENERATOR:
    for max_value in max_values:
        yield create_default_divergent_cmap()
        yield create_default_cmap(value=max_value)


class CalendarFormatError(Exception):
    pass


def plot_calendar_by_row(
    df: pd.DataFrame,
    max_cols: int = 3,
    title_func: TITLE_FUNC | None = None,
    day_labeler: DayLabeler | None = None,
    time_labeler: TimeLabeler | None = None,
    cmaps: CMAP | ColorMap | CMAP_GENERATOR | None = None,
    grid_lines: GridLines = GridLines(),
    monday_start: bool = True,
) -> None:
    """Iterate a DataFrame by row and plot calendar events.

    Args:
        df: wide DataFrame where each column is the vocabulary
        max_cols: max number of columns in the created grid.
        title_func: function to make the title from DataFrame index and DataFrame row, default like '2020-01-01 n_trip(s) = 10'
        day_labeler: base day_labeler
        time_labeler: base day_labeler
        cmaps: Colormapping function(s) to use for each row
        grid_lines: GridLines instance
        monday_start: whether to start the week on Monday or Sunday

    Returns:
        None

    """
    n_cols = len(df.columns)
    if n_cols % 7 != 0:
        raise CalendarFormatError(
            f"Number of columns must be a multiple of 7, got {n_cols} columns. Make sure DataFrame is in wide calendar format."
        )

    title_func = title_func if title_func is not None else default_title_func

    if isinstance(cmaps, ColorMap):
        cmaps = repeat(cmaps)

    if cmaps is None:
        cmaps = repeat(create_default_cmap(value=df.to_numpy().max()))

    total = len(df)

    for (ax, plot_axes), (idx, row), cmap in zip(
        default_axes_and_grid_axes(
            total=total,
            max_cols=max_cols,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        ),
        df.iterrows(),
        cmaps,
    ):
        calendar_data = row.to_numpy()
        plot_calendar(
            iterate_long_array(calendar_data),
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
            ax=ax,
            cmap=cmap,
            monday_start=monday_start,
        )
        title = title_func(idx, row)
        ax.set_title(title)


def plot_dataframe_grid_across_column(
    df: pd.DataFrame,
    grid_col: str,
    config: DataFrameConfig | None = None,
    max_cols: int = 3,
    *,
    alpha: float | None = None,
    monday_start: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
    grid_lines: GridLines = GridLines(),
) -> None:
    """Plot the long DataFrame in a grid by some different column.

    Continuous version of the plot_calendar_by_row

    Args:
        df: DataFrame to plot. Requires all the columns in config
        grid_col: column name of DataFrame to plot across
        config: DataFrameConfig instance of the column mapping. Default IterConfig
        max_cols: max number of columns in the grid
        alpha: alpha of each calendar event
        monday_start: whether to start the week on Monday or Sunday

    """
    if grid_col not in df.columns:
        msg = f"{grid_col} is not in the DataFrame."
        raise KeyError(msg)

    values = df.loc[:, grid_col].dropna().unique()
    # Use np.sort() for pandas 3.0 compatibility (ArrowStringArray has no sort() method)
    values = np.sort(values)

    total = len(values)

    for (ax, plot_axes), value in zip(
        default_axes_and_grid_axes(
            total=total,
            max_cols=max_cols,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        ),
        values,
    ):
        idx = df[grid_col] == value
        df_tmp = df.loc[idx, :]

        day_labeler, time_labeler = plot_axes

        plot_dataframe_as_calendar(
            df=df_tmp,
            config=config,
            ax=ax,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            alpha=alpha,
            monday_start=monday_start,
        )
        ax.set_title(f"{value}")
