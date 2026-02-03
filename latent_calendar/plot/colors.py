"""Handling the colors in the calendar plots.

This module provides some helper function and some defaults. However, they might
not be the best of all purposes.

Color maps here take floats to a color string. Usually a hex string.

Example:
    Create a color map for count data.

    ```python
    cmap = create_default_cmap(max_value=10)

    cmap(0)  # '#ffffe5'
    cmap(5)  # '#379e54'
    cmap(10) # '#004529'
    ```

"""

from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable

import numpy as np

from latent_calendar.plot.config import CONFIG


CM = Callable[[float], tuple[int, int, int, int]]
CMAP = Callable[[float], str]


class ColorMap(ScalarMappable):
    """This supports colorbar for a figure from matplotlib.

    Args:
        norm: matplotlib.colors.Normalize
        cmap: matplotlib.cm.ScalarMappable
        default_cm: matplotlib.cm.ScalarMappable

    """

    def __init__(self, norm, cmap, default_cm: CM) -> None:
        cmap = cmap if cmap is not None else default_cm
        super().__init__(norm=norm, cmap=cmap)

    def __call__(self, x: float) -> str:
        return rgb2hex(self.cmap(self.norm(x)))

    def add_colorbar(self, ax=None) -> None:
        """Add the colorbar to axis or axes.

        Args:
            ax: single or np.ndarray of Axes

        """
        fig = plt.gcf()

        fig.colorbar(self, ax=ax, ticks=self.norm.ticks)


def create_cmap(max_value: float, min_value: float = 0.0, cm: CM | None = None) -> CMAP:
    """Create color map function.

    Args:
        max_value: maximum value for the color map
        min_value: minimum value for the color map
        cm: function that takes a value and returns a color

    """
    norm = Normalize(vmin=min_value, vmax=max_value)
    norm.ticks = [norm.vmin, norm.vmax]
    return ColorMap(norm=norm, cmap=cm, default_cm=plt.cm.YlGn)


def create_diverge_cmap(
    center_value: float, range: float, cm: CM | None = None
) -> CMAP:
    """Create color map function to emphasize a center value and deviation from that center.

    Might be good for values that are relative to some baseline

    """
    half_range = range / 2
    norm = Normalize(vmin=center_value - half_range, vmax=center_value + half_range)
    norm.ticks = [norm.vmin, 0, norm.vmax]
    return ColorMap(norm=norm, cmap=cm, default_cm=plt.cm.coolwarm)


def create_relative_cmap(range: float) -> CMAP:
    """Good for relative scales."""
    return create_diverge_cmap(center_value=1.0, range=range)


# Helper Functions
def create_default_cmap(value: float) -> CMAP:
    max_value = value * CONFIG.max_value_ratio

    return create_cmap(min_value=0.0, max_value=max_value)


def create_default_divergent_cmap() -> CMAP:
    return create_diverge_cmap(center_value=1.0, range=CONFIG.range)


def settle_data_and_cmap(data, divergent: bool) -> tuple[np.ndarray, CMAP]:
    """Return a tuple of transformed data and cmap for displaying that data."""
    if divergent:
        # Comparing the values to random rate
        EVEN_PROBABILITY = 1 / data.shape[0]
        data = data / EVEN_PROBABILITY
        return data, create_default_divergent_cmap()

    return data, create_default_cmap(value=data.max())
