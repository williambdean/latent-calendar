"""Plotting functions for latent calendar.

These functions and classes build every calendar plot.

"""

from latent_calendar.plot.colors import (  # noqa
    create_cmap,
    create_diverge_cmap,
    create_relative_cmap,
    create_default_cmap,
    create_default_divergent_cmap,
    settle_data_and_cmap,
)
from latent_calendar.plot.core import (  # noqa
    plot_blank_calendar,
    plot_calendar,
    plot_dataframe_as_calendar,
    plot_series_as_calendar,
    plot_dataframe_grid_across_column,
    plot_calendar_by_row,
    plot_profile,
    plot_profile_by_row,
    plot_model_predictions,
    plot_model_predictions_by_row,
    plot_raw_data,
    plot_distribution,
    plot_component_distribution,
    plot_model_components,
    plot_component_sensitivity,
)
from latent_calendar.plot.elements import (  # noqa
    CalendarEvent,
    DayLabeler,
    DisplaySettings,
    GridLines,
    HourFormatter,
    TimeLabeler,
    create_default_days,
)
from latent_calendar.plot.grid_settings import (  # noqa
    default_plot_axes_in_grid,
    display_settings_in_grid,
)
from latent_calendar.plot.iterate import (  # noqa
    iterate_dataframe,
    iterate_series,
    iterate_long_array,
    iterate_matrix,
    IterConfig,
    VocabIterConfig,
)
