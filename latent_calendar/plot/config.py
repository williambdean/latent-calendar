"""Configuration file for plotting."""

from dataclasses import dataclass


@dataclass
class Config:
    """Default configuration used in some of the plots.

    Args:
        divergent: whether to show divergent calendar by default.
        range: Where to a divergent plot
        max_value_ratio: Where to clip the default cmap in the calendar view

    """

    divergent: bool = True
    range: float = 3.0
    max_value_ratio: float = 0.75


# The loaded in configuration
CONFIG = Config()
