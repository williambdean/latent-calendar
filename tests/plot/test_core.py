import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from latent_calendar.generate import wide_format_dataframe
from latent_calendar.model.latent_calendar import LatentCalendar
from latent_calendar.plot.core.calendar import plot_calendar
from latent_calendar.plot.core.model import plot_model_components
from latent_calendar.plot.iterate import iterate_matrix


@pytest.fixture
def calendar_data():
    return np.ones((7, 168))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
    ],
)
def test_plot_calendar(calendar_data: pd.DataFrame, kwargs) -> None:
    """Test the plot_calendar function."""
    ax = plot_calendar(
        iterate_matrix(calendar_data),
        **kwargs,
    )

    assert isinstance(ax, plt.Axes)


@pytest.fixture
def mock_data() -> pd.DataFrame:
    return wide_format_dataframe(10, rate=1.0, random_state=42)


@pytest.fixture
def mock_latent_calendar(mock_data) -> LatentCalendar:
    model = LatentCalendar()
    model.fit(mock_data.to_numpy())

    return model


def test_plot_model_components(mock_latent_calendar) -> None:
    """Test the plot_model_components function."""
    plot_model_components(mock_latent_calendar)


def test_plot_model_components_subset(mock_latent_calendar) -> None:
    with pytest.raises(ValueError):
        plot_model_components(
            mock_latent_calendar, components=[mock_latent_calendar.n_components]
        )
