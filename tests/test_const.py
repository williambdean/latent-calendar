import pytest

from latent_calendar.const import dicretized_hours


@pytest.mark.parametrize(
    "minutes, first_five",
    [
        (60, [0, 1, 2, 3, 4]),
        (30, [0, 0.5, 1, 1.5, 2]),
        (15, [0, 0.25, 0.5, 0.75, 1]),
        (120, [0, 2, 4, 6, 8]),
    ],
)
def test_dicretized_hours(minutes: int, first_five: list[float]) -> None:
    assert dicretized_hours(minutes=minutes)[:5] == first_five
