import pytest

import matplotlib.pyplot as plt

from latent_calendar.plot import plot_blank_calendar
from latent_calendar.plot.elements import (
    CalendarEvent,
    DayLabeler,
    TimeLabeler,
    GridLines,
)

from latent_calendar.segments import create_box_segment, stack_segments


@pytest.mark.mpl_image_compare
def test_blank_calendar() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_blank_calendar(ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_various_elements() -> plt.Figure:
    """From the docs: https://williambdean.github.io/latent-calendar/examples/plotting/add-calendar-events/"""
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_blank_calendar(ax=ax)
    event = CalendarEvent(day=0, start=12, duration=15)
    event.plot(ax=ax, label="15 minute event", linestyle="--", alpha=0.25)

    event = CalendarEvent(
        day=0,
        start=23.5,
        end=24.5,
    )
    event.plot(
        ax=ax, label="Two day event", facecolor="red", linestyle="dashed", lw=1.5
    )

    event = CalendarEvent(
        day=5,
        start=11,
        end=17,
    )
    event.plot(ax=ax, label="Friday event", facecolor="green", alpha=0.25)

    ax.legend()
    return fig


@pytest.fixture
def df_segments():
    mornings = create_box_segment(
        day_start=0, day_end=7, hour_start=6, hour_end=11, name="Mornings"
    )
    afternoons = create_box_segment(
        day_start=0, day_end=7, hour_start=11, hour_end=16, name="Afternoons"
    )
    evenings = create_box_segment(
        day_start=0, day_end=7, hour_start=16, hour_end=21, name="Evenings"
    )

    return stack_segments(
        [
            mornings,
            afternoons,
            evenings,
        ]
    )


@pytest.mark.mpl_image_compare
def test_segements(df_segments) -> plt.Figure:
    df_segments.cal.plot_by_row()

    fig = plt.gcf()
    fig.set_size_inches(10, 5)

    return fig


@pytest.mark.mpl_image_compare
def test_settings() -> plt.Figure:
    fig, axes = plt.subplots(ncols=2, nrows=2)
    fig.suptitle("Calendar Customization")

    ax = axes[0, 0]
    plot_blank_calendar(ax=ax)
    ax.set_title("Default")

    ax = axes[0, 1]
    plot_blank_calendar(ax=ax, monday_start=False)
    ax.set_title("Sunday Start")

    ax = axes[1, 0]
    day_labeler = DayLabeler(
        days_of_week=["M", "T", "W", "Th", "F", "Sa", "Su"], rotation=0
    )

    def hour_formatter(hour):
        return f"{hour}h"

    time_labeler = TimeLabeler(hour_formatter=hour_formatter, stride=4)
    plot_blank_calendar(
        ax=ax,
        day_labeler=day_labeler,
        time_labeler=time_labeler,
    )
    ax.set_title("Custom Labels")

    ax = axes[1, 1]
    grid_lines = GridLines(dow=True, hour=True)
    plot_blank_calendar(ax=ax, grid_lines=grid_lines)
    ax.set_title("Adding Grid Lines")

    fig.set_size_inches(15, 15)
    return fig


@pytest.mark.mpl_image_compare
def test_multiday_event() -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_blank_calendar(ax=ax)
    event = CalendarEvent(day=0, start=12, duration=2 * 60, days=3)
    event.plot(ax=ax, label="3 day block", linestyle="--", alpha=0.25, lw=1.5)

    event = CalendarEvent(day=6, start=5, end=7, days=3)
    event.plot(
        ax=ax,
        label="3 day block into next week",
        linestyle="--",
        facecolor="green",
        alpha=0.25,
        lw=1.5,
    )

    event = CalendarEvent(day=6, start=23, duration=2 * 60, days=3)
    event.plot(
        ax=ax,
        label="3 day block into next day and week",
        linestyle="--",
        facecolor="orange",
        alpha=0.25,
        lw=1.5,
    )

    event = CalendarEvent(day=2, start=23, duration=2 * 60, days=2)
    event.plot(
        ax=ax,
        label="2 day block into next day",
        linestyle="--",
        alpha=0.25,
        facecolor="red",
        lw=1.5,
    )

    ax.legend()

    return fig
