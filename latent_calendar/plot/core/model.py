"""Plots including a model."""

from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from latent_calendar.model.latent_calendar import LatentCalendar

from latent_calendar.plot.core.calendar import plot_calendar
from latent_calendar.plot.colors import settle_data_and_cmap, create_default_cmap
from latent_calendar.plot.elements import DayLabeler, TimeLabeler
from latent_calendar.plot.grid_settings import default_axes_and_grid_axes
from latent_calendar.plot.iterate import iterate_long_array


def plot_profile(
    array: np.ndarray,
    model: LatentCalendar,
    divergent: bool = True,
    axes: Iterable[plt.Axes] = None,
    include_components: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> np.ndarray:
    """Create a profile plot with 3 different plots.

    Displays the raw data, predicted probability distribution, and latent breakdown.

    Args:
        array: long array (n_timeslots, )
        model: LatentCalendar model instance
        divergent: Option to change the data displayed
        axes: list of 3 axes to plot this data
        include_components: If the last component plot should be included
        day_labeler: DayLabeler instance
        time_labeler: TimeLabeler instance

    Returns:
        None

    """
    ncols = 3 if include_components else 2
    if axes is None:
        _, axes = plt.subplots(nrows=1, ncols=ncols)

    if len(axes) != ncols:
        msg = "The axes do not equal the number of plots required."
        raise ValueError(msg)

    # Data under model
    X_new = array[np.newaxis, :]
    X_probs = model.predict(X_new)[0]

    # Raw Data
    ax = axes[0]
    plot_raw_data(
        array=array, ax=ax, day_labeler=day_labeler, time_labeler=time_labeler
    )

    # Under Model
    ax = axes[1]
    plot_distribution(
        X_probs=X_probs,
        ax=ax,
        display_y_axis=False,
        divergent=divergent,
        day_labeler=day_labeler,
        time_labeler=time_labeler,
    )

    # Component distribution
    if include_components:
        ax = axes[2]
        X_latent = model.transform(X_new)[0]
        plot_component_distribution(X_latent=X_latent, model=model, ax=ax)

    return axes


def _remove_xticks(axes_row) -> None:
    for ax in axes_row:
        ax.set_xticklabels([])
        ax.set_xlabel("")


def _remove_title(axes_row) -> None:
    for ax in axes_row:
        ax.set_title("")


def _remove_duplicate_labels(axes_row, i: int, nrows: int) -> None:
    if i != nrows - 1:
        _remove_xticks(axes_row)

    if i != 0:
        _remove_title(axes_row)


def plot_profile_by_row(
    df: pd.DataFrame,
    model: LatentCalendar,
    index_func,
    divergent: bool = True,
    include_components: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> np.ndarray:
    nrows = len(df)

    ncols = 3 if include_components else 2
    _, axes_grid = plt.subplots(nrows, ncols)
    if axes_grid.ndim == 1:
        axes_grid = axes_grid[None, :]

    default_stride = 2 if nrows <= 2 else 4
    time_labeler.stride = default_stride

    for i, ((idx, row), axes_row) in enumerate(zip(df.iterrows(), axes_grid)):
        plot_profile(
            row.to_numpy(),
            model=model,
            axes=axes_row,
            divergent=divergent,
            include_components=include_components,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

        ylabel = index_func(idx)
        axes_row[0].set_ylabel(ylabel)

        _remove_duplicate_labels(axes_row, i, nrows)

    return axes_grid


def plot_model_predictions(
    X_to_predict: np.ndarray,
    X_holdout: np.ndarray,
    model: LatentCalendar,
    divergent: bool = True,
    axes: Iterable[plt.Axes] = None,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> Iterable[plt.Axes]:
    """Plot the model predictions compared to the test data.

    Args:
        X_to_predict: Data for the model
        X_holdout: Holdout data for the model
        model: LatentCalendar model instance
        divergent: Option to change the data displayed
        axes: list of 3 axes to plot this data

    Returns:
        The axes used for plotting

    """
    X_to_predict = X_to_predict[np.newaxis, :]
    X_holdout = X_holdout[np.newaxis, :]

    if axes is None:
        _, axes = plt.subplots(nrows=1, ncols=3)

    X_to_predict_probs = model.predict(X_to_predict)[0]

    ax = axes[0]
    plot_raw_data(
        array=X_to_predict, ax=ax, day_labeler=day_labeler, time_labeler=time_labeler
    )
    ax.set_title("Raw Data for Prediction")

    ax = axes[1]
    plot_distribution(
        X_probs=X_to_predict_probs,
        ax=ax,
        display_y_axis=False,
        divergent=divergent,
        day_labeler=day_labeler,
        time_labeler=time_labeler,
    )
    ax.set_title("Distribution from Prediction")

    ax = axes[2]
    plot_raw_data(
        array=X_holdout,
        ax=ax,
        display_y_axis=False,
        day_labeler=day_labeler,
        time_labeler=time_labeler,
    )
    ax.set_title("Raw Data in Future")

    return axes


def plot_model_predictions_by_row(
    df: pd.DataFrame,
    df_holdout: pd.DataFrame,
    model: LatentCalendar,
    index_func=lambda idx: idx,
    divergent: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> np.ndarray:
    nrows = len(df)

    df_holdout = df_holdout.loc[df.index]

    _, axes_grid = plt.subplots(nrows, 3)
    if axes_grid.ndim == 1:
        axes_grid = axes_grid[None, :]

    default_stride = 2 if nrows <= 2 else 4
    time_labeler.stride = default_stride

    for i, ((idx, row), axes_row) in enumerate(zip(df.iterrows(), axes_grid)):
        plot_model_predictions(
            row.to_numpy(),
            df_holdout.loc[idx].to_numpy(),
            model=model,
            axes=axes_row,
            divergent=divergent,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

        ylabel = index_func(idx)
        axes_row[0].set_ylabel(ylabel)

        _remove_duplicate_labels(axes_row, i, nrows)

    return axes_grid


def plot_raw_data(
    array: np.ndarray,
    ax: plt.Axes,
    display_y_axis: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> plt.Axes:
    """First plot of raw data."""
    try:
        max_value = np.quantile(array[array > 0], 0.95)
    except IndexError:
        max_value = 1

    time_labeler.display = display_y_axis

    cmap = create_default_cmap(value=max_value)

    plot_calendar(
        iterate_long_array(array),
        ax=ax,
        cmap=cmap,
        time_labeler=time_labeler,
        day_labeler=day_labeler,
    )
    ax.set_title("Raw Data")

    return ax


def plot_distribution(
    X_probs: np.ndarray,
    ax: plt.Axes,
    display_y_axis: bool = True,
    divergent: bool = True,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> plt.Axes:
    """Second plot of the profile calendar probability distribution."""
    time_labeler.display = display_y_axis

    data, cmap = settle_data_and_cmap(data=X_probs, divergent=divergent)

    iter_data = iterate_long_array(data)

    subtext = "Comparison to random rate" if divergent else "Raw Probabilities"
    plot_calendar(
        iter_data,
        ax=ax,
        cmap=cmap,
        day_labeler=day_labeler,
        time_labeler=time_labeler,
    )
    title = f"Predicted Probability Distribution\n{subtext}"
    ax.set_title(title)

    return ax


def plot_component_sensitivity(X_latent: np.ndarray, counts, ax: plt.Axes) -> plt.Axes:
    count_start, count_end = counts[0], counts[-1]
    n_components = X_latent.shape[1]

    ax.plot(counts, X_latent, label=[f"{i + 1}" for i in range(n_components)])
    ax.set_xlabel("Number of events")
    ax.set_xticks(counts, minor=True)
    ax.set_xlim(count_start, count_end)
    ax.set_ylabel("Component probabilities")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, 0.05), minor=True)
    ax.set_title("Component distribution across event counts")
    if n_components <= 5:
        ax.legend(title="Component")

    return ax


def plot_component_distribution(
    X_latent: np.ndarray, model: LatentCalendar, ax: plt.Axes
) -> plt.Axes:
    """Third profile plot."""
    x = range(len(X_latent))
    ax.bar(x, X_latent)
    step = 1 if model.n_components < 15 else 2
    ax.set_xticks(np.arange(model.n_components, step=step))
    ax.set_ylabel("P[L=l | Data]")
    ax.set_title("Latent Component Distribution")

    return ax


def plot_model_components(
    model: LatentCalendar,
    max_cols: int = 5,
    divergent: bool = True,
    components: Iterable[int] | None = None,
    day_labeler: DayLabeler = DayLabeler(),
    time_labeler: TimeLabeler = TimeLabeler(),
) -> None:
    """Helper function to create plot of all the components of the LatentCalendar instance.

    Args:
        model: LatentCalendar instance
        max_cols: maximum number of columns in the grid of calendar components.
        divergent: what data to plot
        components: Specific subset of components to plot. Default is all
        day_labeler: DayLabeler instance
        time_labeler: TimeLabeler instance

    Returns:
        None

    """
    if components is None:
        components = list(range(model.n_components))

    if any([component > model.n_components - 1 for component in components]):
        msg = f"One of the listed components is greater than the total number {model.n_components}"
        raise ValueError(msg)

    total = len(components)
    normalized_components_to_plot = model.normalized_components_[components]

    def get_title(component_idx: int) -> str:
        return f"Component {component_idx}"

    # TOOD: refactor to just use the plot_calendar_by_row ?
    values = zip(
        components,
        default_axes_and_grid_axes(
            total=total,
            max_cols=max_cols,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        ),
        normalized_components_to_plot,
    )
    for component_idx, (ax, plot_axes), latent in values:
        data, cmap = settle_data_and_cmap(latent, divergent)

        day_labeler, time_labeler = plot_axes
        plot_calendar(
            iterate_long_array(data),
            cmap=cmap,
            ax=ax,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )
        title = get_title(component_idx=component_idx)
        ax.set_title(title)
