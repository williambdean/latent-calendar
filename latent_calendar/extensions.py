"""DataFrame extensions for `latent-calendar` and primary interface for the package.

!!! tip

    The extensions work for both pandas and polars DataFrames. However, there is currently
    limited functionality for polars DataFrames. Consider using polars for aggregation then
    converting to pandas for plotting and model training.

Provides a `cal` accessor to `DataFrame` and `Series` instances for easy transformation and plotting after import of `latent_calendar`.

Functionality includes:

- aggregation of events to wide format
- convolutions of wide formats
- making transformations and predictions with models
- plotting of events, predictions, and comparisons as calendars

Each `DataFrame` will be either at event level or an aggregated wide format.

Methods that end in `row` or `by_row` will be for wide format DataFrames and will plot each row as a calendar.

Examples:
    Plotting an event level Series as a calendar

    ```python
    import pandas as pd
    import latent_calendar

    dates = pd.date_range("2023-01-01", "2023-01-14", freq="h")
    ser = (
        pd.Series(dates)
        .sample(10, random_state=42)
    )

    ser.cal.plot()
    ```

    ![Series Calendar](./../images/series-calendar.png)

    Transform event level DataFrame to wide format and plot

    ```python
    from latent_calendar.datasets import load_online_transactions

    df = load_online_transactions()

    # (n_customer, n_timeslots)
    df_wide = (
        df
        .cal.aggregate_events("Customer ID", timestamp_col="InvoiceDate")
    )

    (
        df_wide
        .sample(n=12, random_state=42)
        .cal.plot_by_row(max_cols=4)
    )
    ```

    ![Customer Transactions](./../images/customer-transactions.png)

    Train a model and plot predictions

    ```python
    from latent_calendar import LatentCalendar

    model = LatentCalendar(n_components=5, random_state=42)
    model.fit(df_wide.to_numpy())

    (
        df_wide
        .head(2)
        .cal.plot_profile_by_row(model=model)
    )
    ```

    ![Profile By Row](./../images/profile-by-row.png)


"""

from typing import Literal

import narwhals as nw

import pandas as pd
import numpy as np

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import matplotlib.pyplot as plt

from latent_calendar.model.latent_calendar import LatentCalendar
from latent_calendar.model.utils import transform_on_dataframe, predict_on_dataframe
from latent_calendar.plot.colors import CMAP, ColorMap
from latent_calendar.plot.core import (
    plot_calendar_by_row,
    plot_profile_by_row,
    plot_dataframe_as_calendar,
    plot_series_as_calendar,
    plot_dataframe_grid_across_column,
    plot_model_predictions_by_row,
)
from latent_calendar.plot.core.calendar import TITLE_FUNC, CMAP_GENERATOR
from latent_calendar.plot.elements import DayLabeler, TimeLabeler, GridLines
from latent_calendar.plot.iterate import StartEndConfig

from latent_calendar.segments.convolution import (
    sum_over_segments,
    sum_over_vocab,
    sum_next_hours,
)
from latent_calendar.transformers import (
    create_raw_to_vocab_transformer,
    create_timestamp_feature_pipeline,
    LongToWide,
    raw_to_aggregate,
    create_timestamp_features,
    create_discretized_hour,
    create_vocab,
)


@pd.api.extensions.register_series_accessor("cal")
class PandasSeriesAccessor:
    """Series accessor for latent_calendar accessed through `cal` attribute of Series."""

    def __init__(self, pandas_obj: pd.Series):
        self._obj = pandas_obj

    def aggregate_events(
        self,
        minutes: int = 60,
        as_multiindex: bool = True,
    ) -> pd.Series:
        """Transform event level Series to row of wide format.

        Args:
            minutes: The number of minutes to discretize by.
            as_multiindex: whether to use MultiIndex columns

        Returns:
            Series that would be row of wide format

        Examples:
            Discretize datetime Series to 30 minutes

            ```python
            import pandas as pd

            import matplotlib.pyplot as plt

            from latent_calendar.datasets import load_chicago_bikes

            df_trips = load_chicago_bikes()

            start_times = df_trips["started_at"]

            agg_start_times = start_times.cal.aggregate_events(minutes=30)
            agg_start_times.cal.plot_row()
            plt.show()


            ```


        """
        name = self._obj.name or "timestamp"
        return (
            self._obj.rename(name)
            .to_frame()
            .assign(tmp=1)
            .cal.aggregate_events(
                by="tmp",
                timestamp_col=name,
                minutes=minutes,
                as_multiindex=as_multiindex,
            )
            .iloc[0]
            .rename(name)
        )

    def timestamp_features(
        self,
        discretize: bool = True,
        minutes: int = 60,
        create_vocab: bool = True,
    ) -> pd.DataFrame:
        """Create day of week and proportion into day columns.

        Exposed as a method on Series for convenience.

        Args:
            discretize: Whether to discretize the hour column.
            minutes: The number of minutes to discretize by. Ingored if `discretize` is False.
            create_vocab: Whether to create the vocab column.

        Returns:
            DataFrame with features

        Examples:
            Create the features for some dates

            ```python
            ser = pd.Series(pd.date_range("2023-01-01", "2023-01-14", freq="h"))

            ser.cal.timestamp_features()
            ```

            ```text
                        timestamp  day_of_week  hour
            0   2023-01-01 00:00:00            6   0.0
            1   2023-01-01 01:00:00            6   1.0
            2   2023-01-01 02:00:00            6   2.0
            3   2023-01-01 03:00:00            6   3.0
            4   2023-01-01 04:00:00            6   4.0
            ..                  ...          ...   ...
            308 2023-01-13 20:00:00            4  20.0
            309 2023-01-13 21:00:00            4  21.0
            310 2023-01-13 22:00:00            4  22.0
            311 2023-01-13 23:00:00            4  23.0
            312 2023-01-14 00:00:00            5   0.0

            [313 rows x 3 columns]
            ```

        """
        name = self._obj.name or "timestamp"
        transformer = create_timestamp_feature_pipeline(
            timestamp_col=name,
            discretize=discretize,
            minutes=minutes,
            create_vocab=create_vocab,
        )

        return transformer.fit_transform(self._obj.rename(name).to_frame())

    def conditional_probabilities(
        self,
        *,
        level: int | str = 0,
    ) -> pd.Series:
        """Calculate conditional probabilities for each the row over the level.

        Args:
            level: level of the column MultiIndex.
                Default 0 or day_of_week

        Returns:
            Series with conditional probabilities

        """

        if not isinstance(self._obj.index, pd.MultiIndex):
            raise ValueError(
                "Series is expected to have a MultiIndex with the last column as the vocab."
            )

        return self._obj.div(self._obj.groupby(level=level).sum(), level=level)

    def plot(
        self,
        *,
        duration: int = 5,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot Series of timestamps as a calendar.

        Args:
            duration: duration of each event in minutes
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            day_labeler: DayLabeler instance
            time_labeler: TimeLabeler instance
            grid_lines: GridLines instance
            monday_start: whether to start the week on Monday or Sunday
            ax: matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        tmp_name = "tmp_name"
        config = StartEndConfig(start=tmp_name, end=None, minutes=duration)

        return plot_dataframe_as_calendar(
            self._obj.rename(tmp_name).to_frame(),
            config=config,
            alpha=alpha,
            cmap=cmap,
            monday_start=monday_start,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
            ax=ax,
        )

    def plot_row(
        self,
        *,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot Series of timestamps as a calendar.

        Args:
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            monday_start: whether to start the week on Monday or Sunday
            ax: matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        return plot_series_as_calendar(
            self._obj,
            alpha=alpha,
            cmap=cmap,
            ax=ax,
            monday_start=monday_start,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
        )


@pd.api.extensions.register_dataframe_accessor("cal")
class PandasDataFrameAccessor:
    """DataFrame accessor for latent_calendar accessed through `cal` attribute of DataFrames."""

    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def divide_by_max(self) -> pd.DataFrame:
        """Divide each row by the max value.

        Use before plotting to normalize the values to [0, 1].

        Returns:
            DataFrame with row-wise operations applied

        """
        return self._obj.div(self._obj.max(axis=1), axis=0)

    def divide_by_sum(self) -> pd.DataFrame:
        """Divide each row by the sum of the row.

        Use to create an empirical probability distribution for each row.

        Returns:
            DataFrame with row-wise operations applied

        """
        return self._obj.div(self._obj.sum(axis=1), axis=0)

    def divide_by_even_rate(self) -> pd.DataFrame:
        """Divide each row by the number of columns.

        Use to create a relative rate compared to an even distribution. Greater than
        1 indicates the row has more events than expected under an even distribution.

        Returns:
            DataFrame with row-wise operations applied

        """
        value = self._obj.shape[1]
        return self._obj.mul(value)

    def conditional_probabilities(
        self,
        *,
        level: int | str = 0,
    ) -> pd.DataFrame:
        """Calculate conditional probabilities for each row over the level.

        Args:
            level: level of the columns MultiIndex.
                Default 0 or day_of_week

        Returns:
            DataFrame with conditional probabilities

        """
        if not isinstance(self._obj.columns, pd.MultiIndex):
            raise ValueError(
                "DataFrame is expected to have a MultiIndex with the last column as the vocab."
            )

        return self._obj.div(
            self._obj.T.groupby(level=level).sum().T, level=level, axis=1
        )

    def timestamp_features(
        self,
        column: str,
        discretize: bool = True,
        minutes: int = 60,
        create_vocab: bool = True,
    ) -> pd.DataFrame:
        """Create day of week and proportion into day columns for event level DataFrame

        Exposed as a method on DataFrame for convenience. Use `cal.aggregate_events` instead to create the wide format DataFrame.

        Args:
            column: The name of the timestamp column.
            discretize: Whether to discretize the hour column.
            minutes: The number of minutes to discretize by. Ingored if `discretize` is False.
            create_vocab: Whether to create the vocab column.

        Returns:
            DataFrame with features added

        """
        transformer = create_timestamp_feature_pipeline(
            timestamp_col=column,
            discretize=discretize,
            create_vocab=create_vocab,
            minutes=minutes,
        )

        return transformer.fit_transform(self._obj)

    def widen(
        self,
        column: str,
        as_int: bool = True,
        minutes: int = 60,
        multiindex: bool = True,
    ) -> pd.DataFrame:
        """Transform an aggregated DataFrame to wide calendar format.

        Wrapper around `LongToWide` transformer to transform to wide format.

        Args:
            column: column to widen
            as_int: whether to cast the column to int
            minutes: number of minutes to
            multiindex: whether to use a MultiIndex

        Returns:
            DataFrame in wide format

        """
        if not isinstance(self._obj.index, pd.MultiIndex):
            raise ValueError(
                "DataFrame is expected to have a MultiIndex with the last column as the vocab."
            )

        transformer = LongToWide(
            col=column, as_int=as_int, minutes=minutes, multiindex=multiindex
        )

        return transformer.fit_transform(self._obj)

    def aggregate_events(
        self,
        by: str | list[str],
        timestamp_col: str,
        minutes: int = 60,
        as_multiindex: bool = True,
    ) -> pd.DataFrame:
        """Transform event level DataFrame to wide format with groups as index.

        Wrapper around `create_raw_to_vocab_transformer` to transform to wide format.

        Args:
            by: column(s) to use as index
            timestamp_col: column to use as timestamp
            minutes: The number of minutes to discretize by.
            as_multiindex: whether to use MultiIndex columns

        Returns:
            DataFrame in wide format

        """
        if not isinstance(by, list):
            id_col = by
            additional_groups = None
        else:
            id_col, *additional_groups = by

        transformer = create_raw_to_vocab_transformer(
            id_col=id_col,
            timestamp_col=timestamp_col,
            minutes=minutes,
            additional_groups=additional_groups,
            as_multiindex=as_multiindex,
        )
        return transformer.fit_transform(self._obj)

    def sum_over_vocab(
        self,
        aggregation: Literal["dow", "hour"] = "dow",
    ) -> pd.DataFrame:
        """Sum the wide format to day of week or hour of day.

        Args:
            aggregation: one of ['dow', 'hour']

        Returns:
            DataFrame with summed values

        Examples:
            Sum to day of week

            ```python
            df_dow = df_wide.cal.sum_over_vocab(aggregation='dow')
            ```

        """
        return sum_over_vocab(self._obj, aggregation=aggregation)

    def sum_next_hours(self, hours: int) -> pd.DataFrame:
        """Sum the wide format over next hours.

        Args:
            hours: number of hours to sum over

        Returns:
            DataFrame with summed values

        """
        return sum_next_hours(self._obj, hours=hours)

    def sum_over_segments(self, df_segments: pd.DataFrame) -> pd.DataFrame:
        """Sum the wide format over user defined segments.

        Args:
            df_segments: DataFrame in wide format with segments as index

        Returns:
            DataFrame with columns as the segments and summed values

        """
        return sum_over_segments(self._obj, df_segments=df_segments)

    def transform(self, *, model: LatentCalendar) -> pd.DataFrame:
        """Transform DataFrame with model.

        Applies the dimensionality reduction to each row of the DataFrame.

        Args:
            model: model to use for transformation

        Returns:
            DataFrame with transformed values

        """
        return transform_on_dataframe(self._obj, model=model)

    def predict(self, *, model: LatentCalendar) -> pd.DataFrame:
        """Predict DataFrame with model.

        Args:
            model: model to use for prediction

        Returns:
            DataFrame with predicted values (wide format)

        """
        return predict_on_dataframe(self._obj, model=model)

    def plot(
        self,
        start_col: str,
        *,
        end_col: str | None = None,
        duration: int | None = None,
        alpha: float = None,
        cmap=None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot DataFrame of timestamps as a calendar.

        Args:
            start_col: column with start timestamp
            end_col: column with end timestamp
            duration: length of event in minutes. Alternative to end_col
            alpha: alpha value for the color
            cmap: function that maps floats to string colors
            monday_start: whether to start the week on Monday or Sunday
            ax: optional matplotlib axis to plot on

        Returns:
            Modified matplotlib axis

        """
        config = StartEndConfig(start=start_col, end=end_col, minutes=duration)

        return plot_dataframe_as_calendar(
            self._obj,
            config=config,
            alpha=alpha,
            cmap=cmap,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
            monday_start=monday_start,
            ax=ax,
        )

    def plot_across_column(
        self,
        start_col: str,
        grid_col: str,
        *,
        end_col: str | None = None,
        duration: int | None = None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        max_cols: int = 3,
        alpha: float = None,
    ) -> None:
        """Plot DataFrame of timestamps as a calendar as grid across column values.

        NA values are excluded

        Args:
            start_col: column with start timestamp
            grid_col: column of values to use as grid
            end_col: column with end timestamp
            duration: length of event in minutes. Alternative to end_col
            max_cols: max number of columns per row
            alpha: alpha value for the color

        Returns:
            None

        """
        config = StartEndConfig(start=start_col, end=end_col, minutes=duration)

        plot_dataframe_grid_across_column(
            self._obj,
            grid_col=grid_col,
            config=config,
            max_cols=max_cols,
            alpha=alpha,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            grid_lines=grid_lines,
        )

    def plot_by_row(
        self,
        *,
        max_cols: int = 3,
        title_func: TITLE_FUNC | None = None,
        cmaps: CMAP | ColorMap | CMAP_GENERATOR | None = None,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
        grid_lines: GridLines = GridLines(),
        monday_start: bool = True,
    ) -> None:
        """Plot each row of the DataFrame as a calendar plot. Data must have been transformed to wide format first.

        Wrapper around `latent_calendar.plot.plot_calendar_by_row`.

        Args:
            max_cols: max number of columns per row of grid
            title_func: function to generate title for each row
            day_labeler: function to generate day labels
            time_labeler: function to generate time labels
            cmaps: optional generator of colormaps
            grid_lines: optional grid lines
            monday_start: whether to start the week on Monday or Sunday

        Returns:
            None

        """
        return plot_calendar_by_row(
            self._obj,
            max_cols=max_cols,
            title_func=title_func,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
            cmaps=cmaps,
            grid_lines=grid_lines,
            monday_start=monday_start,
        )

    def plot_profile_by_row(
        self,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        include_components: bool = True,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot each row of the DataFrame as a profile plot. Data must have been transformed to wide format first.

        Args:
            model: model to use for prediction and transform
            index_func: function to generate title for each row
            include_components: whether to include components in the plot
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_profile_by_row(
            self._obj,
            model=model,
            index_func=index_func,
            include_components=include_components,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

    def plot_raw_and_predicted_by_row(
        self,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot raw and predicted values for a model. Data must have been transformed to wide format first.

        Args:
            model: model to use for prediction
            index_func: function to generate title for each row
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_profile_by_row(
            self._obj,
            model=model,
            index_func=index_func,
            include_components=False,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

    def plot_model_predictions_by_row(
        self,
        df_holdout: pd.DataFrame,
        *,
        model: LatentCalendar,
        index_func=lambda idx: idx,
        divergent: bool = True,
        day_labeler: DayLabeler = DayLabeler(),
        time_labeler: TimeLabeler = TimeLabeler(),
    ) -> np.ndarray:
        """Plot model predictions for each row of the DataFrame. Data must have been transformed to wide format first.

        Args:
            df_holdout: holdout DataFrame for comparison
            model: model to use for prediction
            index_func: function to generate title for each row
            divergent: whether to use divergent colormap
            day_labeler: DayLabeler instance to use for day labels
            time_labeler: TimeLabeler instance to use for time labels

        Returns:
            grid of axes

        """
        return plot_model_predictions_by_row(
            self._obj,
            df_holdout=df_holdout,
            model=model,
            index_func=index_func,
            divergent=divergent,
            day_labeler=day_labeler,
            time_labeler=time_labeler,
        )

    def chart(
        self,
        *,
        title: str | None = None,
        width: int = 400,
        height: int = 300,
        color_scheme: str = "greens",
        monday_start: bool = True,
        interactive: bool = True,
        show_values: bool = True,
    ):
        """Create an Altair calendar heatmap chart.

        Convenience wrapper around create_calendar_chart() that works directly
        on the DataFrame without requiring an import.

        Works with:

        - Multi-row wide format (from .cal.aggregate_events())
        - Single-row wide format (168 columns)
        - Long format (day_of_week, hour, value columns)

        Returns Altair Chart that can be:

        - Displayed in notebooks
        - Saved with .save('file.html')
        - Faceted with .facet(column='group:N')

        Args:
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            color_scheme: Altair color scheme ('greens', 'blues', 'viridis', etc.)
            monday_start: Start week on Monday (True) or Sunday (False)
            interactive: Enable tooltips and zoom/pan
            show_values: Show values in tooltips

        Returns:
            Altair Chart object

        Examples:
            Create chart from aggregated data:

            ```python
            chart = df.cal.aggregate_events("group", "timestamp").cal.chart()
            chart.save('calendar.html')
            ```

            Faceted chart in one line:

            ```python
            (df.cal.aggregate_events("member_casual", "started_at")
               .cal.chart(color_scheme="greens")
               .facet(column='member_casual:N'))
            ```

        """
        from latent_calendar.html import create_calendar_chart

        return create_calendar_chart(
            self._obj,
            title=title,
            width=width,
            height=height,
            color_scheme=color_scheme,
            monday_start=monday_start,
            interactive=interactive,
            show_values=show_values,
        )


if HAS_POLARS:

    @pl.api.register_dataframe_namespace("cal")
    class PolarsDataFrameAccessor:
        """Polars extension accessor for latent_calendar.

        Examples:

        Register the accessor on a Polars DataFrame and use it to aggregate events.

        ```python
        import polars as pl

        # Register the accessor
        import latent_calendar

        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
        df = pl.read_parquet(url)

        df_agg = df.cal.aggregate_events(
            "PULocationID",
            "tpep_pickup_datetime",
        )
        df_agg
        ```

        ```text
        shape: (32_051, 4)
        ┌──────────────┬─────────────┬──────┬────────────┐
        │ PULocationID ┆ day_of_week ┆ hour ┆ num_events │
        │ ---          ┆ ---         ┆ ---  ┆ ---        │
        │ i32          ┆ i8          ┆ i64  ┆ i32        │
        ╞══════════════╪═════════════╪══════╪════════════╡
        │ 76           ┆ 2           ┆ 15   ┆ 16         │
        │ 143          ┆ 0           ┆ 21   ┆ 123        │
        │ 153          ┆ 4           ┆ 7    ┆ 2          │
        │ 18           ┆ 3           ┆ 20   ┆ 2          │
        │ 100          ┆ 4           ┆ 11   ┆ 350        │
        │ …            ┆ …           ┆ …    ┆ …          │
        │ 178          ┆ 6           ┆ 1    ┆ 1          │
        │ 180          ┆ 2           ┆ 14   ┆ 3          │
        │ 82           ┆ 6           ┆ 2    ┆ 2          │
        │ 151          ┆ 0           ┆ 5    ┆ 28         │
        │ 67           ┆ 2           ┆ 13   ┆ 2          │
        └──────────────┴─────────────┴──────┴────────────┘
        ```

        """

        def __init__(self, polars_obj):
            self._obj = polars_obj

        def timestamp_features(
            self,
            timestamp_col: str,
            discretize: bool = True,
            minutes: int = 60,
            create_vocab: bool = True,
        ):
            """Create day of week and proportion into day columns for event level DataFrame.

            Exposed as a method on Polars DataFrame for convenience. Use `cal.aggregate_events` instead to create the wide format DataFrame.

            Args:
                timestamp_col: The name of the timestamp column.
                discretize: Whether to discretize the hour column.
                minutes: The number of minutes to discretize by. Ingored if `discretize` is False.
                create_vocab: Whether to create the vocab column.

            Returns:
                DataFrame with features added

            """
            transformer = create_timestamp_feature_pipeline(
                timestamp_col=timestamp_col,
                discretize=discretize,
                create_vocab=create_vocab,
                minutes=minutes,
                output="polars",
            )

            return transformer.fit_transform(self._obj)

        def aggregate_events(
            self,
            by: str | list[str],
            timestamp_col: str,
            minutes: int = 60,
            as_multiindex: bool = True,
        ):
            """Transform event level Polars DataFrame to aggregated.

            !!! note

                Wide format is not supported in Polars yet.

            Args:
                by: column(s) to use as index
                timestamp_col: column to use as timestamp
                minutes: The number of minutes to discretize by.
                as_multiindex: whether to use MultiIndex columns

            Returns:
                DataFrame in wide format

            """
            return create_raw_to_vocab_transformer(
                id_col=by if isinstance(by, str) else by[0],
                timestamp_col=timestamp_col,
                minutes=minutes,
                additional_groups=None if isinstance(by, str) else by[1:],
                as_multiindex=as_multiindex,
                widen=False,
            ).fit_transform(self._obj)

        def chart(
            self,
            *,
            title: str | None = None,
            width: int = 400,
            height: int = 300,
            color_scheme: str = "greens",
            monday_start: bool = True,
            interactive: bool = True,
            show_values: bool = True,
        ):
            """Create an Altair calendar heatmap chart from Polars DataFrame.

            Convenience wrapper that converts Polars DataFrame to pandas
            and creates an Altair chart.

            Note: Polars .cal.aggregate_events() returns long format with
            (by_column, day_of_week, hour, num_events) columns which is
            automatically handled by create_calendar_chart().

            Args:
                title: Chart title
                width: Chart width in pixels
                height: Chart height in pixels
                color_scheme: Altair color scheme ('greens', 'blues', 'viridis', etc.)
                monday_start: Start week on Monday (True) or Sunday (False)
                interactive: Enable tooltips and zoom/pan
                show_values: Show values in tooltips

            Returns:
                Altair Chart object

            Examples:
                Create chart from Polars aggregated data:

                ```python
                import polars as pl

                chart = df.cal.aggregate_events("group", "timestamp").cal.chart()
                chart.save('calendar.html')
                ```

                Faceted chart:

                ```python
                (df.cal.aggregate_events("member_casual", "started_at")
                   .cal.chart(color_scheme="greens")
                   .facet(column='member_casual:N'))
                ```

            """
            from latent_calendar.html import create_calendar_chart

            # Pass directly to create_calendar_chart - it handles Polars via narwhals
            return create_calendar_chart(
                self._obj,
                title=title,
                width=width,
                height=height,
                color_scheme=color_scheme,
                monday_start=monday_start,
                interactive=interactive,
                show_values=show_values,
            )

    @pl.api.register_lazyframe_namespace("cal")
    class PolarsLazyFrameAccessor:
        """LazyFrame extension accessor for latent_calendar.

        Examples:

        ```python
        import polars as pl

        # Register the accessor
        import latent_calendar

        url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet"
        df = pl.read_parquet(url).lazy()

        df_agg = df.cal.aggregate_events(
            "PULocationID",
            "tpep_pickup_datetime",
        )
        df_agg.collect()
        ```

        ```text
        shape: (32_051, 4)
        ┌──────────────┬─────────────┬──────┬────────────┐
        │ PULocationID ┆ day_of_week ┆ hour ┆ num_events │
        │ ---          ┆ ---         ┆ ---  ┆ ---        │
        │ i32          ┆ i8          ┆ i64  ┆ i32        │
        ╞══════════════╪═════════════╪══════╪════════════╡
        │ 207          ┆ 1           ┆ 8    ┆ 2          │
        │ 232          ┆ 2           ┆ 10   ┆ 19         │
        │ 97           ┆ 2           ┆ 9    ┆ 11         │
        │ 74           ┆ 5           ┆ 10   ┆ 49         │
        │ 92           ┆ 1           ┆ 6    ┆ 2          │
        │ …            ┆ …           ┆ …    ┆ …          │
        │ 1            ┆ 4           ┆ 16   ┆ 12         │
        │ 34           ┆ 3           ┆ 14   ┆ 1          │
        │ 74           ┆ 1           ┆ 23   ┆ 24         │
        │ 102          ┆ 0           ┆ 5    ┆ 1          │
        │ 212          ┆ 0           ┆ 10   ┆ 3          │
        └──────────────┴─────────────┴──────┴────────────┘
        ```


        """

        def __init__(self, obj):
            self._obj = obj

        def timestamp_features(
            self,
            timestamp_col: str,
            minutes: int = 60,
        ) -> pl.LazyFrame:
            """Create day of week and proportion into day columns for event level LazyFrame."""
            return (
                nw.from_native(self._obj)
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
                .to_native()
            )

        def aggregate_events(
            self,
            by: str | list[str],
            timestamp_col: str,
            minutes: int = 60,
        ) -> pl.LazyFrame:
            """Aggregate the event level Polars LazyFrame to aggregated format."""
            return self._obj.pipe(
                raw_to_aggregate,
                id_col=by if isinstance(by, str) else by[0],
                timestamp_col=timestamp_col,
                minutes=minutes,
                additional_groups=None if isinstance(by, str) else by[1:],
            )
