"""scikit-learn transformers for the data.

```python
from latent_calendar.datasets import load_online_transactions

df = load_online_transactions()

transformers = create_raw_to_vocab_transformer(id_col="Customer ID", timestamp_col="InvoiceDate")

df_wide = transformers.fit_transform(df)
```


"""

import warnings

import narwhals as nw
from narwhals.typing import FrameT

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from latent_calendar.const import (
    create_full_vocab,
    DAYS_IN_WEEK,
    HOURS_IN_DAY,
    MINUTES_IN_DAY,
    SECONDS_IN_DAY,
    MICROSECONDS_IN_DAY,
)


def prop_into_day(dt: nw.expr_dt.ExprDateTimeNamespace) -> nw.Expr:
    """Returns the proportion into the day from datetime like object.

    0.0 is midnight and 1.0 is midnight again.

    Args:
        dt: datetime like object

    Returns:
        numeric value(s) between 0.0 and 1.0

    """
    prop_hour = dt.hour() / HOURS_IN_DAY
    prop_minute = dt.minute() / MINUTES_IN_DAY
    prop_second = dt.second() / SECONDS_IN_DAY
    prop_microsecond = dt.microsecond() / MICROSECONDS_IN_DAY

    return prop_hour + prop_minute + prop_second + prop_microsecond


@nw.narwhalify
def create_timestamp_features(df: FrameT, timestamp_col: str) -> FrameT:
    col = nw.col(timestamp_col)

    prop_into_day_start = prop_into_day(col.dt)
    day_of_week = col.dt.weekday()

    return df.with_columns(
        day_of_week=day_of_week,
        hour=prop_into_day_start * HOURS_IN_DAY,
    )


class CalandarTimestampFeatures(BaseEstimator, TransformerMixin):
    """Day of week and prop into day columns creation."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def __init__(
        self,
        timestamp_col: str,
    ) -> None:
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    @nw.narwhalify
    def transform(self, X, y=None):
        """Create 2 new columns."""

        X = create_timestamp_features(X, self.timestamp_col)
        X.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


class HourDiscretizer(BaseEstimator, TransformerMixin):
    """Discretize the hour column.

    Args:
        col: The name of the column to discretize.
        minutes: The number of minutes to discretize by.

    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def __init__(self, col: str = "hour", minutes: int = 60) -> None:
        self.col = col
        self.minutes = minutes

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        divisor = 1 if self.minutes == 60 else self.minutes / 60
        X[self.col] = (X[self.col] // divisor) * divisor

        if self.minutes % 60 == 0:
            X[self.col] = X[self.col].astype(int)

        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


class VocabTransformer(BaseEstimator, TransformerMixin):
    """Create a vocab column from the day of week and hour columns."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def __init__(
        self, day_of_week_col: str = "day_of_week", hour_col: str = "hour"
    ) -> None:
        self.day_of_week_col = day_of_week_col
        self.hour_col = hour_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X["vocab"] = (
            X[self.day_of_week_col]
            .astype(str)
            .str.zfill(2)
            .str.cat(X[self.hour_col].astype(str).str.zfill(2), sep=" ")
        )

        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


def create_timestamp_feature_pipeline(
    timestamp_col: str,
    discretize: bool = True,
    minutes: int = 60,
    create_vocab: bool = True,
) -> Pipeline:
    """Create a pipeline that creates features from the timestamp column.

    Args:
        timestamp_col: The name of the timestamp column.
        discretize: Whether to discretize the hour column.
        minutes: The number of minutes to discretize by. Ignored if discretize is False.
        create_vocab: Whether to create the vocab column.

    Returns:
        A pipeline that creates features from the timestamp column.

    Example:
        Create features for the online transactions dataset.

        ```python
        from latent_calendar.datasets import load_online_transactions

        df = load_online_transactions()

        transformers = create_timestamp_feature_pipeline(timestamp_col="InvoiceDate")

        df_features = transformers.fit_transform(df)
        ```

    """
    if create_vocab and not discretize:
        raise ValueError("Cannot create vocab without discretizing.")

    vocab_col = "hour"
    transformers = [
        (
            "timestamp_features",
            CalandarTimestampFeatures(timestamp_col=timestamp_col),
        ),
    ]

    if discretize:
        transformers.append(
            ("binning", HourDiscretizer(col=vocab_col, minutes=minutes))
        )

    if create_vocab:
        transformers.append(
            ("vocab_creation", VocabTransformer(hour_col=vocab_col)),
        )

    return Pipeline(
        transformers,
    ).set_output(transform="pandas")


class VocabAggregation(BaseEstimator, TransformerMixin):
    """NOTE: The index of the grouping stays.

    Args:
        groups: The columns to group by.
        cols: Additional columns to sum.

    """

    def __init__(self, groups: list[str], cols: list[str] | None = None) -> None:
        self.groups = groups
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        stats = {}
        if self.cols is not None:
            stats.update({col: (col, "sum") for col in self.cols})

        df_agg = (
            X.assign(num_events=1)
            .groupby(self.groups)
            .agg(num_events=("num_events", "sum"), **stats)
        )
        self.columns = list(df_agg.columns)

        return df_agg

    def get_feature_names_out(self, input_features=None):
        return self.columns


class LongToWide(BaseEstimator, TransformerMixin):
    """Unstack the assumed last index as vocab column.

    Args:
        col: The name of the column to unstack.
        as_int: Whether to cast the values to int.
        minutes: The number of minutes to discretize by.
        multiindex: Whether the columns are a multiindex.

    """

    def __init__(
        self,
        col: str = "num_events",
        as_int: bool = True,
        minutes: int = 60,
        multiindex: bool = True,
    ) -> None:
        self.col = col
        self.as_int = as_int
        self.minutes = minutes
        self.multiindex = multiindex

    def fit(self, X: pd.DataFrame, y=None):
        return self

    @property
    def columns(self) -> list[str]:
        return create_full_vocab(
            days_in_week=DAYS_IN_WEEK,
            minutes=self.minutes,
            as_multiindex=self.multiindex,
        )

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Unstack the assumed last index as vocab column."""
        X_res = X.loc[:, self.col]

        level = [-2, -1] if self.multiindex else -1
        X_res = X_res.unstack(level=level)

        X_res = X_res.reindex(self.columns, axis=1)
        X_res = X_res.fillna(value=0)
        if self.as_int:
            X_res = X_res.astype(int)

        return X_res

    def get_feature_names_out(self, input_features=None):
        return self.columns


class RawToVocab(BaseEstimator, TransformerMixin):
    """Transformer timestamp level data into id level data with vocab columns.

    Args:
        id_col: The name of the id column.
        timestamp_col: The name of the timestamp column.
        minutes: The number of minutes to discretize by.
        additional_groups: Additional columns to group by.
        cols: Additional columns to sum.
        as_multiindex: Whether to return columns as a multiindex.

    """

    def __init__(
        self,
        id_col: str,
        timestamp_col: str,
        minutes: int = 60,
        additional_groups: list[str] | None = None,
        cols: list[str] | None = None,
        as_multiindex: bool = True,
    ) -> None:
        self.id_col = id_col
        self.timestamp_col = timestamp_col
        self.minutes = minutes
        self.additional_groups = additional_groups
        self.cols = cols
        self.as_multiindex = as_multiindex

    def fit(self, X: pd.DataFrame, y=None):
        # New features at same index level
        self.features = create_timestamp_feature_pipeline(
            self.timestamp_col,
            minutes=self.minutes,
            create_vocab=not self.as_multiindex,
        )

        groups = [self.id_col]
        if self.additional_groups is not None:
            if not isinstance(self.additional_groups, list):
                raise ValueError(
                    f"additional_groups should be list not {type(self.additional_groups)}"
                )

            groups.extend(self.additional_groups)

        if self.as_multiindex:
            groups.extend(["day_of_week", "hour"])
        else:
            groups.append("vocab")

        # Reaggregation
        self.aggregation = VocabAggregation(groups=groups, cols=self.cols)
        # Unstacking
        self.widden = LongToWide(
            col="num_events", minutes=self.minutes, multiindex=self.as_multiindex
        )
        # Since nothing needs to be "fit"
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_trans = self.features.transform(X)

        X_agg = self.aggregation.transform(X_trans)
        return self.widden.transform(X_agg)


def create_raw_to_vocab_transformer(
    id_col: str,
    timestamp_col: str,
    minutes: int = 60,
    additional_groups: list[str] | None = None,
    as_multiindex: bool = True,
) -> RawToVocab:
    """Wrapper to create the transformer from the configuration options.

    Args:
        id_col: The name of the id column.
        timestamp_col: The name of the timestamp column.
        minutes: The number of minutes to discretize by.
        additional_groups: Additional columns to group by.
        as_multiindex: Whether to return columns as a multiindex.

    Returns:
        A transformer that transforms timestamp level data into id level data with vocab columns.

    """
    if not as_multiindex:
        msg = (
            "columns will be returned as a MultiIndex by default and will "
            "be behavior in future. Use as_multiindex=False for previous behavior"
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    return RawToVocab(
        id_col=id_col,
        timestamp_col=timestamp_col,
        minutes=minutes,
        additional_groups=additional_groups,
    )
