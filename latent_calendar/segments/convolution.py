"""Processing off calendar distribution."""

from typing import Literal

import pandas as pd
import numpy as np


def _reverse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse the order of the columns."""
    return df.iloc[:, ::-1]


def sum_next_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Sum the next hours columns.

    Useful for finding probability of having tour in the next 5 hours
    00 00 column would be 06 06 23

    TODO: Consider if negative hours should be allowed
    TODO: Handle when minutes are not 60

    Arguments:
        df: DataFrame of probabilities or counts in wide format
        hours: Number of hours to sum after the current hour

    Returns:
        DataFrame summed over the next hours

    """
    if hours < 0:
        msg = "hours cannot be negative"
        raise ValueError(msg)

    if hours == 0:
        return df

    return (
        pd.concat([df, df.iloc[:, :hours]], axis=1)
        .pipe(_reverse_columns)
        .T.rolling(hours + 1)
        .sum()
        .T.iloc[:, hours:]
        .pipe(_reverse_columns)
    )


def _mask_probs(X_segments, X_pred) -> np.ndarray:
    """Multiply out the mask.

    Args:
        X_segments: (n_segments, n_times)
        X_pred: (nrows, n_times)

    Returns:
        (n_segments, nrows, n_times) matrix of only the values that fall into the segments times

    """
    return X_segments[:, None, :] * X_pred


def sum_array_over_segments(X_pred: np.ndarray, X_segment: np.ndarray) -> np.ndarray:
    """Get the probability of the mask for the probabilities.

    Args:
        X_pred: (nrows, n_times)
        X_segment: (n_segments, n_times)

    Returns:
        Matrix of (nrows, n_segments) defining the probabilities of each segments

    """
    return _mask_probs(X_segment, X_pred).sum(axis=2).T


def sum_over_segments(df: pd.DataFrame, df_segments: pd.DataFrame) -> pd.DataFrame:
    """Sum DataFrame over user defined segments.

    Args:
        df: DataFrame of probabilities or counts in wide format
        df_segments: DataFrame of segments in wide format

    Returns:
        DataFrame of probabilities or counts summed over the segments

    """
    return pd.DataFrame(
        sum_array_over_segments(df.to_numpy(), df_segments.to_numpy()),
        index=df.index,
        columns=df_segments.index,
    )


def sum_over_vocab(
    df: pd.DataFrame,
    aggregation: Literal["dow", "hour"] = "dow",
) -> pd.DataFrame:
    """Sum the wide DataFrame columns to hours or dow.

    Args:
        df: DataFrame in wide format with vocab column names
        aggregation: either dow or hour

    Returns:
        DataFrame columns associated with the aggregation

    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("The columns must be a MultiIndex of day_of_week and hour.")

    level = 1 if aggregation == "hour" else 0
    return df.T.groupby(level=level).sum().T
