from typing import Generator

from matplotlib import gridspec
import matplotlib.pyplot as plt

from latent_calendar.plot.elements import (
    DayLabeler,
    DisplaySettings,
    TimeLabeler,
)


def col_idx(i: int, ncols: int) -> int:
    return i % ncols


def is_left_edge(i: int, ncols: int) -> bool:
    return col_idx(i, ncols) == 0


def row_idx(i: int, ncols: int) -> int:
    return i // ncols


def last_in_column(i: int, nrows: int, ncols: int, total: int) -> bool:
    grid_size = nrows * ncols
    difference = grid_size - total

    if difference > ncols:
        msg = f"{nrows = } and {ncols = } with {total = }. This combination doesn't make sense. One whole row will be empty!"
        raise ValueError(msg)

    # Past the last values requires the second to last row instead of last
    offset = 2 if col_idx(i, ncols) >= ncols - difference else 1

    return row_idx(i, ncols) == (nrows - offset)


def display_settings_in_grid(
    nrows: int,
    ncols: int,
    total: int | None = None,
) -> Generator[DisplaySettings, None, None]:
    """Helper for display logic in a grid.

    Can be used with zip since zip function will stop at the shorts of the iterators

    Yields:
        DisplaySettings instance with the appropriate settings based on position in the grid.

    """
    total = total if total is not None else nrows * ncols

    yield from (
        DisplaySettings(
            x=last_in_column(i, nrows, ncols, total), y=is_left_edge(i, ncols)
        )
        for i in range(total)
    )


PlotAxes = tuple[DayLabeler, TimeLabeler]


def default_plot_axes_in_grid(
    nrows: int,
    ncols: int,
    total: int | None = None,
    day_labeler: DayLabeler | None = None,
    time_labeler: TimeLabeler | None = None,
) -> Generator[PlotAxes, None, None]:
    """Additional layer on the display_settings_in_grid in order to modify the settings.

    Yields:
        PlotAxes instance with appropriate display settings based on the position in the grid.

    """
    day_labeler = day_labeler if day_labeler is not None else DayLabeler()
    default_stride = 2 if nrows <= 2 else 4
    time_labeler = (
        time_labeler if time_labeler is not None else TimeLabeler(stride=default_stride)
    )

    for display_settings in display_settings_in_grid(
        nrows=nrows, ncols=ncols, total=total
    ):
        day_labeler.display = display_settings.x
        time_labeler.display = display_settings.y

        yield day_labeler, time_labeler


def grid_axes(nrows: int, ncols: int, total: int) -> Generator[plt.Axes, None, None]:
    """Yields a grid of size nrow, ncols with total cap.

    Using this instead of plt.subplots(ncols, nrows) and deleting

    """
    gs = gridspec.GridSpec(nrows, ncols)

    fig = plt.figure()

    yield from (fig.add_subplot(gs[i]) for i in range(total))


def get_rows_and_cols(n: int, max_cols: int) -> tuple[int, int]:
    """Return the number of rows and cols."""
    nrows = max((n // max_cols) + 1, 1)
    ncols = min(n, max_cols)

    if n % max_cols == 0:
        nrows -= 1

    return nrows, ncols


def default_axes_and_grid_axes(
    total: int,
    max_cols: int,
    day_labeler: DayLabeler | None = None,
    time_labeler: TimeLabeler | None = None,
) -> Generator[tuple[plt.Axes, PlotAxes], None, None]:
    nrows, ncols = get_rows_and_cols(n=total, max_cols=max_cols)
    yield from zip(
        grid_axes(nrows=nrows, ncols=ncols, total=total),
        default_plot_axes_in_grid(
            nrows=nrows,
            ncols=ncols,
            total=total,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        ),
    )
