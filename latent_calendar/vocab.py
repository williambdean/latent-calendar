"""Operations and relationship with the "vocab" of the default time slots."""

from dataclasses import dataclass
import calendar
from typing import Callable

import pandas as pd

from latent_calendar.const import HOURS_IN_DAY, format_dow_hour


def am_pm_of_hour(hour: int) -> str:
    """Get the am or pm of the hour.

    Args:
        hour: hour of the day

    Returns:
        am or pm

    """
    return "am" if hour < 12 else "pm"


def map_to_12_hour(hour: int) -> int:
    """Map the hour to a 12 hour clock."""
    if hour == 0:
        return 12

    if hour == 12:
        return hour

    return hour % 12


HOUR_FORMATTER = Callable[[int], str]
HOUR_FORMATTERS: dict[str, HOUR_FORMATTER] = {
    "24hr": lambda hr: f"{hr} O'Clock",
    "12_am_pm": lambda hr: f"{map_to_12_hour(hr)}{am_pm_of_hour(hr)}",
    "12hr": lambda hr: f"{map_to_12_hour(hr)} O'Clock",
    "de": lambda hr: f"{hr} Uhr",
}


@dataclass
class HourFormatter:
    """Class to format the hour that includes midnight and noon.

    Args:
        midnight: string to use for midnight
        noon: string to use for noon
        format_hour: HOUR_FORMATTER to map hour int to string

    Examples:
        Just return the number and add midnight and noon.

        ```python
        hour_formatter = HourFormatter(
            midnight="Midnight",
            noon="Noon",
            format_hour=lambda hour: hour
        )

        hour_formatter(0) # "Midnight"
        hour_formatter(12) # "Noon"
        hour_formatter(1) # 1
        hour_formatter(13) # 13
        hour_formatter(24) # "Midnight"

        ```

    """

    midnight: str | None = "Midnight"
    noon: str | None = "Noon"
    format_hour: HOUR_FORMATTER = HOUR_FORMATTERS["12hr"]

    def __call__(self, hr: int) -> str:
        if self.midnight is not None and hr in (0, HOURS_IN_DAY):
            return self.midnight

        if hr == 12:
            return self.noon if self.noon is not None else self.format_hour(hr)

        return self.format_hour(hr)


def get_day_hour(vocab: str) -> tuple[int, int]:
    """Get the day and hour from the vocab."""
    day_str, hour_str = vocab.split(" ")

    return int(day_str), int(hour_str)


def make_human_readable(
    vocab: str, hour_formatter: HOUR_FORMATTER = HOUR_FORMATTERS["12_am_pm"]
) -> str:
    """Create a human readable string of the vocab.

    Args:
        vocab: string vocab. i.e. "00 01"
        hour_formatter: HOUR_FORMATTER to map hour int to string

    Returns:
        human readable string of the vocab

    """
    day, hour = get_day_hour(vocab=vocab)

    human_day = calendar.day_name[day]
    human_hour = hour_formatter(hour)

    return f"{human_day} {human_hour}"


@dataclass
class DOWHour:
    """Day of week and hour of day class."""

    dow: int
    hour: int

    @classmethod
    def from_vocab(cls, vocab: str) -> "DOWHour":
        """Construct from a vocab string."""
        dow, hour = get_day_hour(vocab=vocab)

        return cls(dow=dow, hour=hour)

    def __post_init__(self) -> None:
        msg = "Day of week goes from 0 to 6 and hour of day goes from 0 to 24."
        if not 0 <= self.dow <= 6:
            raise ValueError(msg)

        if not 0 <= self.hour <= 24:
            raise ValueError(msg)

    def is_after(self, other: "DOWHour") -> bool:
        """Check if self is after other."""
        if self.dow > other.dow:
            return True

        if self.dow < other.dow:
            return False

        return self.hour > other.hour

    @property
    def vocab(self) -> str:
        """Get the vocab string for an instance."""
        return format_dow_hour(self.dow, self.hour)

    def __add__(self, hours: int) -> "DOWHour":
        """Add a number of hours."""
        dow, hour = self.dow, self.hour

        for _ in range(hours):
            hour += 1
            if hour > 23:
                dow += 1
                dow = dow % 7

            hour = hour % 24

        return DOWHour(dow=dow, hour=hour)


def split_vocab(ser: pd.Series) -> pd.DataFrame:
    """Split pandas series of vocab into day of week and hour of day DataFrame."""
    df_split = ser.str.split(" ", expand=True).astype(int)

    df_split.columns = ["dow", "hour"]

    return df_split
