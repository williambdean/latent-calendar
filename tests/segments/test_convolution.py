import numpy as np
import pandas as pd
import pytest

from latent_calendar.generate import wide_format_dataframe
from latent_calendar.segments.convolution import (
    sum_array_over_segments,
    sum_next_hours,
    sum_over_vocab,
)


@pytest.mark.parametrize(
    "data, hours, answer",
    [
        ([[1, 2, 3, 4]], 0, [[1, 2, 3, 4]]),
        ([[1, 2, 3, 4]], 1, [[3, 5, 7, 5]]),
        ([[1, 2, 3, 4]], 2, [[6, 9, 8, 7]]),
        ([[1, 2, 3, 4]], 3, [[10, 10, 10, 10]]),
    ],
)
def test_sum_previous_hours(data, hours, answer) -> None:
    df = pd.DataFrame(data).astype(float)
    df_answer = pd.DataFrame(answer).astype(float)

    df_result = sum_next_hours(df, hours)
    assert df.shape == df_result.shape
    pd.testing.assert_frame_equal(df_result, df_answer)


@pytest.mark.parametrize(
    "X_strat, X, answer",
    [
        (
            # Strategies
            [[1, 0, 1], [0, 1, 0]],
            # Users
            [[0, 1, 1], [1, 2, 2], [1, 2, 3], [0.5, 0.5, 0]],
            [[1, 1], [3, 2], [4, 2], [0.5, 0.5]],
        )
    ],
)
def test_sum_over_strategies(X_strat, X, answer) -> None:
    X_strat = np.array(X_strat)
    X = np.array(X)

    answer = np.array(answer)

    result = sum_array_over_segments(X, X_strat)

    n_strats, _ = X_strat.shape
    n_users, _ = X.shape

    assert result.shape == (n_users, n_strats)

    np.testing.assert_allclose(result, answer)


@pytest.fixture
def df_wide() -> pd.DataFrame:
    return wide_format_dataframe(n_rows=10, random_state=42)


@pytest.mark.parametrize(
    "aggregation, num_columns",
    [
        ("dow", 7),
        ("hour", 24),
    ],
)
def test_sum_over_vocab(
    df_wide: pd.DataFrame, aggregation: str, num_columns: int
) -> None:
    df_result = df_wide.pipe(sum_over_vocab, aggregation=aggregation)

    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (len(df_wide), num_columns)
