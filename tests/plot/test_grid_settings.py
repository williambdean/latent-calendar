import pytest

import numpy as np

from latent_calendar.plot.grid_settings import (
    last_in_column,
    is_left_edge,
    get_rows_and_cols,
)


@pytest.mark.parametrize(
    "nrows, ncols, answer",
    [
        (2, 2, [[True, False], [True, False]]),
        (
            3,
            4,
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
            ],
        ),
    ],
)
def test_is_left_edge(nrows: int, ncols: int, answer: list[list[bool]]) -> None:
    answer = np.array(answer)

    grid_size = nrows * ncols
    result = np.array([is_left_edge(i, ncols) for i in range(grid_size)]).reshape(
        nrows, ncols
    )

    np.testing.assert_array_equal(result, answer)


@pytest.mark.parametrize(
    "nrows, ncols, total, answer",
    [
        (
            2,
            2,
            3,
            [
                [False, True],
                [True, False],
            ],
        ),
        (2, 3, 4, [[False, True, True], [True, False, False]]),
        (
            3,
            3,
            7,
            [
                [False, False, False],
                [False, True, True],
                [True, False, False],
            ],
        ),
        (
            3,
            3,
            8,
            [
                [False, False, False],
                [False, False, True],
                [True, True, False],
            ],
        ),
        (
            3,
            3,
            9,
            [
                [False, False, False],
                [False, False, False],
                [True, True, True],
            ],
        ),
    ],
)
def test_last_in_column(
    nrows: int, ncols: int, total: int, answer: list[list[bool]]
) -> None:
    answer = np.array(answer)
    grid_size = nrows * ncols
    result = np.array(
        [last_in_column(i, nrows, ncols, total) for i in range(grid_size)]
    ).reshape(nrows, ncols)

    assert result.sum() == ncols
    np.testing.assert_array_equal(result, answer)


@pytest.mark.parametrize(
    "n, max_cols, answer",
    [
        # Two rows until one row
        (6, 3, (2, 3)),
        (5, 3, (2, 3)),
        (4, 3, (2, 3)),
        (3, 3, (1, 3)),
        # Less than max
        (2, 3, (1, 2)),
        (1, 3, (1, 1)),
    ],
)
def test_get_rows_and_col(n: int, max_cols: int, answer: int) -> None:
    result = get_rows_and_cols(n=n, max_cols=max_cols)

    assert result == answer
