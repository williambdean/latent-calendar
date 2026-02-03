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
from narwhals.typing import FrameT, IntoFrameT

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
    if not isinstance(dt, nw.expr_dt.ExprDateTimeNamespace):
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        microsecond = dt.microsecond
    else:
        hour = dt.hour()
        minute = dt.minute()
        second = dt.second()
        microsecond = dt.microsecond()

    prop_hour = hour / HOURS_IN_DAY
    prop_minute = minute / MINUTES_IN_DAY
    prop_second = second / SECONDS_IN_DAY
    prop_microsecond = microsecond / MICROSECONDS_IN_DAY

    return prop_hour + prop_minute + prop_second + prop_microsecond


def create_timestamp_features(df: FrameT, timestamp_col: str) -> FrameT:
    col = nw.col(timestamp_col)

    prop_into_day_start = prop_into_day(col.dt)
    day_of_week = col.dt.weekday() - 1

    return df.with_columns(
        day_of_week=day_of_week,
        hour=prop_into_day_start * HOURS_IN_DAY,
    )


class CalendarTimestampFeatures(BaseEstimator, TransformerMixin):
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

        X = create_timestamp_features(X, self.timestamp_col).to_native()
        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


def CalandarTimestampFeatures(*arg, **kwargs) -> CalendarTimestampFeatures:
    """Alias for CalendarTimestampFeatures.

    This is to avoid breaking changes in the API.

    """
    warnings.warn(
        "CalandarTimestampFeatures is deprecated. Use CalendarTimestampFeatures instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CalendarTimestampFeatures(*arg, **kwargs)


def create_discretized_hour(
    X: FrameT,
    col: str = "hour",
    minutes: int = 60,
) -> FrameT:
    divisor = 1 if minutes == 60 else minutes / 60
    name = col
    col = nw.col(col)

    col = (col // divisor) * divisor

    if minutes % 60 == 0:
        col = col.cast(nw.Int64)

    col = col.alias(name)

    return X.with_columns(col)


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

    def fit(self, X, y=None):
        return self

    @property
    def divisor(self) -> float:
        return 1 if self.minutes == 60 else self.minutes / 60

    @nw.narwhalify
    def transform(self, X: FrameT, y=None) -> FrameT:
        X = create_discretized_hour(X, col=self.col, minutes=self.minutes)

        self.columns = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns


def create_vocab(X: FrameT, hour_col: str, day_of_week_col) -> FrameT:
    """Create vocabulary column from day of week and hour.

    Note: We use conditional logic instead of str.zfill(2) because narwhals' str.zfill()
    is not supported for PySpark backend (as of narwhals 2.16.0). This workaround
    specifically handles width=2 padding (sufficient for day_of_week 0-6 and hour 0-23).

    tracking issue: [GitHub Issue](https://github.com/narwhals-dev/narwhals/issues/3442)

    Equivalent to:
        day_str = nw.col(day_of_week_col).cast(nw.String).str.zfill(2)
        hour_str = nw.col(hour_col).cast(nw.String).str.zfill(2)
    """
    # Pad day_of_week to width 2: 0 -> "00", 1 -> "01", ..., 6 -> "06"
    day_of_week_part = (
        nw.when(nw.col(day_of_week_col) < 10)
        .then(nw.concat_str([nw.lit("0"), nw.col(day_of_week_col).cast(nw.String)]))
        .otherwise(nw.col(day_of_week_col).cast(nw.String))
    )

    # Pad hour to width 2: 0 -> "00", 1 -> "01", ..., 23 -> "23"
    hour_part = (
        nw.when(nw.col(hour_col) < 10)
        .then(nw.concat_str([nw.lit("0"), nw.col(hour_col).cast(nw.String)]))
        .otherwise(nw.col(hour_col).cast(nw.String))
    )

    vocab = nw.concat_str([day_of_week_part, hour_part], separator=" ").alias("vocab")

    return X.with_columns(vocab)


class VocabTransformer(BaseEstimator, TransformerMixin):
    """Create a vocab column from the day of week and hour columns."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def __init__(
        self,
        day_of_week_col: str = "day_of_week",
        hour_col: str = "hour",
    ) -> None:
        self.day_of_week_col = day_of_week_col
        self.hour_col = hour_col

    @nw.narwhalify
    def fit(self, X, y=None):
        self.columns = X.columns + ["vocab"]
        return self

    @nw.narwhalify
    def transform(self, X: FrameT, y=None) -> FrameT:
        return create_vocab(
            X,
            hour_col=self.hour_col,
            day_of_week_col=self.day_of_week_col,
        )

    def get_feature_names_out(self, input_features=None):
        return self.columns


def create_timestamp_feature_pipeline(
    timestamp_col: str,
    discretize: bool = True,
    minutes: int = 60,
    create_vocab: bool = True,
    output: str = "pandas",
) -> Pipeline:
    """Create a pipeline that creates features from the timestamp column.

    Args:
        timestamp_col: The name of the timestamp column.
        discretize: Whether to discretize the hour column.
        minutes: The number of minutes to discretize by. Ignored if discretize is False.
        create_vocab: Whether to create the vocab column.
        output: The output type of the pipeline. Default is "pandas"

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
            CalendarTimestampFeatures(timestamp_col=timestamp_col),
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
    ).set_output(transform=output)


def aggregate_vocab(
    X: FrameT,
    groups: list[str],
    cols: list[str] | None = None,
) -> FrameT:
    stats = []
    if cols is not None:
        stats = [nw.col(col).sum() for col in cols]

    return (
        X.with_columns(num_events=nw.lit(1))
        .group_by(groups)
        .agg(
            [
                nw.col("num_events").sum(),
                *stats,
            ]
        )
        .pipe(nw.maybe_set_index, column_names=groups)
    )


class VocabAggregation(BaseEstimator, TransformerMixin):
    """NOTE: The index of the grouping stays for pandas DataFrames.

    Args:
        groups: The columns to group by.
        cols: Additional columns to sum.

    """

    def __init__(self, groups: list[str], cols: list[str] | None = None) -> None:
        self.groups = groups
        self.cols = cols

    def fit(self, X, y=None):
        self.columns = [
            *self.groups,
            *(self.cols or []),
            "num_events",
        ]
        return self

    @nw.narwhalify
    def transform(self, X: FrameT, y=None):
        return aggregate_vocab(X, self.groups, cols=self.cols)

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


def raw_to_aggregate(
    df: IntoFrameT,
    id_col: str,
    timestamp_col: str,
    minutes: int = 60,
    additional_groups: list[str] | None = None,
    cols: list[str] | None = None,
) -> IntoFrameT:
    """Aggregate raw timestamp level data into

    This function uses narwhals and will work on any supported DataFrame implementation.

    Args:
        df: The input data.
        id_col: The name of the id column.
        timestamp_col: The name of the timestamp column.
        minutes: The number of minutes to discretize by.
        additional_groups: Additional columns to group by.
        cols: Additional columns to sum.

    Returns:
        A DataFrame with aggregated data.

    Example:
        Aggregate DataFrame in a polars LazyFrame

        ```python
        import polars as pl

        from latent_calendar.datasets import load_online_transactions
        from latent_calendar import raw_to_aggregate

        df = load_online_transactions()
        df_lazy = pl.LazyFrame(df)

        df_agg = raw_to_aggregate(
            df=df_lazy,
            id_col="Country",
            timestamp_col="InvoiceDate",
        )

        df_agg.collect()
        ```

        ```
        shape: (1_088, 4)
        ┌────────────────┬─────────────┬──────┬────────────┐
        │ Country        ┆ day_of_week ┆ hour ┆ num_events │
        │ ---            ┆ ---         ┆ ---  ┆ ---        │
        │ str            ┆ i8          ┆ i64  ┆ i32        │
        ╞════════════════╪═════════════╪══════╪════════════╡
        │ Belgium        ┆ 2           ┆ 15   ┆ 1          │
        │ Germany        ┆ 0           ┆ 8    ┆ 112        │
        │ EIRE           ┆ 4           ┆ 16   ┆ 18         │
        │ Italy          ┆ 0           ┆ 11   ┆ 1          │
        │ Canada         ┆ 4           ┆ 12   ┆ 1          │
        │ …              ┆ …           ┆ …    ┆ …          │
        │ Finland        ┆ 3           ┆ 19   ┆ 17         │
        │ Australia      ┆ 1           ┆ 14   ┆ 8          │
        │ Portugal       ┆ 1           ┆ 11   ┆ 23         │
        │ United Kingdom ┆ 0           ┆ 11   ┆ 17949      │
        │ Iceland        ┆ 2           ┆ 14   ┆ 29         │
        └────────────────┴─────────────┴──────┴────────────┘
        ```

    """
    return (
        nw.from_native(df)
        .pipe(
            create_timestamp_features,
            timestamp_col=timestamp_col,
        )
        .pipe(
            create_discretized_hour,
            col="hour",
            minutes=minutes,
        )
        .pipe(
            create_vocab,
            hour_col="hour",
            day_of_week_col="day_of_week",
        )
        .pipe(
            aggregate_vocab,
            groups=[id_col, "day_of_week", "hour", *(additional_groups or [])],
            cols=cols,
        )
        .to_native()
    )


class RawToVocab(BaseEstimator, TransformerMixin):
    """Transformer timestamp level data into id level data with vocab columns.

    Args:
        id_col: The name of the id column.
        timestamp_col: The name of the timestamp column.
        minutes: The number of minutes to discretize by.
        additional_groups: Additional columns to group by.
        cols: Additional columns to sum.
        as_multiindex: Whether to return columns as a multiindex.
        widen: Whether to widen the data at the end. Only supported for DataFrames with index.

    """

    def __init__(
        self,
        id_col: str,
        timestamp_col: str,
        minutes: int = 60,
        additional_groups: list[str] | None = None,
        cols: list[str] | None = None,
        as_multiindex: bool = True,
        widen: bool = True,
    ) -> None:
        self.id_col = id_col
        self.timestamp_col = timestamp_col
        self.minutes = minutes
        self.additional_groups = additional_groups
        self.cols = cols
        self.as_multiindex = as_multiindex
        self.widen = widen

    @nw.narwhalify
    def fit(self, X: FrameT, y=None):
        # New features at same index level
        self.features = create_timestamp_feature_pipeline(
            self.timestamp_col,
            minutes=self.minutes,
            create_vocab=not self.as_multiindex,
            output=str(X.implementation),
        )
        self.features.fit(X)

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
        self.aggregation.fit(X)
        if not self.widen:
            return self

        # Unstacking
        self.widen_transformer = LongToWide(
            col="num_events",
            minutes=self.minutes,
            multiindex=self.as_multiindex,
        )
        # Since nothing needs to be "fit"
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_trans = self.features.transform(X)
        X_agg = self.aggregation.transform(X_trans)

        if not self.widen:
            return X_agg

        return self.widen_transformer.transform(X_agg)


def create_raw_to_vocab_transformer(
    id_col: str,
    timestamp_col: str,
    minutes: int = 60,
    additional_groups: list[str] | None = None,
    as_multiindex: bool = True,
    widen: bool = True,
) -> RawToVocab:
    """Wrapper to create the transformer from the configuration options.

    Args:
        id_col: The name of the id column.
        timestamp_col: The name of the timestamp column.
        minutes: The number of minutes to discretize by.
        additional_groups: Additional columns to group by.
        as_multiindex: Whether to return columns as a multiindex.
        widen: Whether to widen the data at the end. Only supported for DataFrames with index.

    Returns:
        A transformer that transforms timestamp level data into id level data with vocab columns.

    """

    return RawToVocab(
        id_col=id_col,
        timestamp_col=timestamp_col,
        minutes=minutes,
        additional_groups=additional_groups,
        as_multiindex=as_multiindex,
        widen=widen,
    )
