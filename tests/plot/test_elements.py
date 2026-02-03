import pytest

from latent_calendar.const import DAYS_IN_WEEK
from latent_calendar.plot.elements import CalendarEvent, TimeLabeler, DayLabeler


@pytest.mark.parametrize(
    "day_start, answer",
    [
        (0, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
        (6, ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]),
    ],
)
def test_day_start(day_start: int, answer: list[str]) -> None:
    day_labeler = DayLabeler(day_start=day_start)
    assert day_labeler.day_labels == answer


@pytest.mark.parametrize(
    "stride, answer",
    [
        (
            2,
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        ),
        (
            4,
            [0, 4, 8, 12, 16, 20, 24],
        ),
        (3, [0, 3, 6, 9, 12, 15, 18, 21, 24]),
    ],
)
def test_hours(stride: int, answer) -> None:
    time_labeler = TimeLabeler(stride=stride)

    assert list(time_labeler.get_hours()) == answer


@pytest.mark.parametrize(
    "start, end, answer",
    [
        (23, 24, False),
        (23, 24.5, True),
        (23, 0.5, True),
        (12, 13, False),
    ],
)
def test_multiday_tour(start: float, end: float, answer: bool) -> None:
    for day in range(DAYS_IN_WEEK):
        event = CalendarEvent(day=day, start=start, end=end)

        assert event.multiday_tour == answer


@pytest.mark.parametrize(
    "start, end, answer",
    [
        (23, 24, [CalendarEvent(day=6, start=23, end=24)]),
        (
            23,
            24.5,
            [
                CalendarEvent(day=6, start=23, end=24),
                CalendarEvent(day=0, start=0, end=0.5),
            ],
        ),
        (12, 12.5, [CalendarEvent(day=6, start=12, end=12.5)]),
    ],
)
def test_separate_events(start: float, end: float, answer: list[CalendarEvent]):
    event = CalendarEvent(
        day=6,
        start=start,
        end=end,
    )

    assert event.separate_events() == answer


@pytest.mark.parametrize("vocab", ["00 01", "00 00", "01 01"])
def test_vocab_contructor(vocab: str) -> None:
    event = CalendarEvent.from_vocab(vocab)
    assert isinstance(event, CalendarEvent)


def test_week_overlap() -> None:
    event = CalendarEvent(day=6, start=5, end=7, days=2)

    assert event.multiweek_tour

    events = event.separate_events()
    assert events == [
        CalendarEvent(day=6, start=5, end=7, days=1),
        CalendarEvent(day=0, start=5, end=7, days=1),
    ]


def test_day_and_week_overlap() -> None:
    event = CalendarEvent(day=6, start=23, duration=2 * 60, days=3)

    assert event.multiweek_tour
    assert event.multiday_tour

    assert event.separate_events() == [
        CalendarEvent(day=6, start=23, end=24, days=1),
        CalendarEvent(day=0, start=0, end=1, days=3),
        CalendarEvent(day=0, start=23, end=24, days=2),
    ]


@pytest.mark.parametrize("day", [-1, 7, 8])
@pytest.mark.parametrize("days", [-1, 0, 8, 9])
def test_calendar_event_init_errors(day, days) -> None:
    with pytest.raises(ValueError):
        CalendarEvent(day=day, start=0, end=1, days=days)
