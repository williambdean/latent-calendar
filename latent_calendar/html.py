"""HTML calendar generation using Altair for interactive visualizations.

This module provides functions to create interactive calendar heatmaps using Altair
and export them as HTML. These calendars can be embedded in web pages, Jupyter notebooks,
or used with mapping libraries like Folium.

Note: This module requires altair to be installed:

```bash
pip install latent-calendar[html]
```

Examples:
    Create a basic calendar chart:

    ```python
    import pandas as pd
    import numpy as np
    from latent_calendar.html import create_calendar_chart

    # Generate sample calendar data (168 time slots: 7 days × 24 hours)
    calendar_data = pd.Series(np.random.rand(168))

    # Create interactive Altair chart
    chart = create_calendar_chart(
        calendar_data,
        title="Weekly Pattern",
        color_scheme='greens'
    )

    # Save as HTML file
    chart.save('calendar.html')
    ```

    Create faceted chart directly from cal.aggregate_events:

    ```python
    from latent_calendar.datasets import load_chicago_bikes
    from latent_calendar.html import create_calendar_chart

    # Load data and aggregate by group
    df = load_chicago_bikes()
    df_agg = df.cal.aggregate_events("member_casual", "started_at")

    # Works directly - automatically converts multi-row wide format!
    chart = create_calendar_chart(df_agg, width=250, height=200, color_scheme='viridis')

    # Apply faceting using Altair's .facet() method
    faceted = chart.facet(column='member_casual:N')
    faceted.save('faceted_calendar.html')
    ```

    Chain methods for compact code:

    ```python
    # Create chart with custom properties and faceting in one expression
    faceted = (
        create_calendar_chart(df_agg, color_scheme='viridis')
        .properties(width=250, height=200, title="Bike Share Patterns")
        .facet(column='member_casual:N', columns=2)
    )
    faceted.save('faceted_calendar.html')
    ```

    Generate HTML string using Altair's to_html():

    ```python
    # Get HTML string for embedding
    html = chart.to_html()

    # Or with custom embed options
    html = chart.to_html(embed_options={'actions': False})

    # Write to file or embed in application
    with open('calendar.html', 'w') as f:
        f.write(html)
    ```
"""

import pandas as pd
import numpy as np
import narwhals as nw

from latent_calendar.plot.iterate import iterate_long_array

try:
    import altair as alt
except ImportError as e:
    raise ImportError(
        "HTML calendar generation requires altair to be installed. "
        "Install it directly or with: pip install latent-calendar[html]"
    ) from e


def wide_to_long_format(calendar_data, monday_start: bool = True) -> pd.DataFrame:
    """Convert wide calendar format (168 cols) to long format for Altair.

    Transforms a calendar in wide format (7 days × 24 hours = 168 values) into
    a long-format DataFrame suitable for Altair visualization with columns for
    day_of_week, hour, and value.

    This function leverages the existing iterate_long_array() generator from
    the plot.iterate module, demonstrating how dataclasses can be easily converted
    to DataFrames.

    Args:
        calendar_data: Series or array-like with 168 values (7 days × 24 hours).
                      Can be pandas Series, numpy array, or list.
        monday_start: Start week on Monday (True) or Sunday (False). Default is True.

    Returns:
        DataFrame with columns: day_of_week (int 0-6), hour (int 0-23), value (float)

    Raises:
        ValueError: If calendar_data does not have exactly 168 values

    Examples:
        ```python
        import pandas as pd
        import numpy as np
        from latent_calendar.html import wide_to_long_format

        # Create sample calendar data
        calendar_data = pd.Series(np.random.rand(168))

        # Convert to long format
        df_long = wide_to_long_format(calendar_data)
        print(df_long.head())
        #    day_of_week  hour     value
        # 0            0     0  0.417022
        # 1            0     1  0.720324
        # 2            0     2  0.000114
        # 3            0     3  0.302333
        # 4            0     4  0.146756
        ```

        With Sunday start:

        ```python
        df_long = wide_to_long_format(calendar_data, monday_start=False)
        ```
    """

    # Convert to numpy array
    values = np.array(calendar_data).ravel()

    # Validate 168 values for Altair hourly visualization
    if len(values) != 168:
        raise ValueError(
            f"Expected 168 values (7 days × 24 hours), got {len(values)}. "
            f"Make sure calendar_data is in wide format with 168 time slots."
        )

    # Use existing iterator to create dataclass objects, then convert to DataFrame
    # CalendarData has (day, start, end, value) fields
    df = pd.DataFrame(iterate_long_array(values))

    # Rename columns for Altair: day -> day_of_week, start -> hour
    # For hourly data, start is already 0, 1, 2, ..., 23 (integers as floats)
    df = df.rename(columns={"day": "day_of_week", "start": "hour"})

    # Drop 'end' column (not needed for Altair)
    df = df[["day_of_week", "hour", "value"]]

    # Convert hour to int (it's whole numbers for hourly data)
    df["hour"] = df["hour"].astype(int)

    # Adjust for Sunday start if needed
    if not monday_start:
        # Shift: Mon(0)→Tue(1), ..., Sat(5)→Sun(6), Sun(6)→Mon(0)
        df["day_of_week"] = (df["day_of_week"] + 1) % 7

    return df


def create_calendar_chart(
    calendar_data,
    *,
    title: str | None = None,
    width: int = 400,
    height: int = 300,
    color_scheme: str = "greens",
    monday_start: bool = True,
    interactive: bool = True,
    show_values: bool = True,
) -> alt.Chart:
    """Create an Altair calendar heatmap chart.

    Generates an interactive calendar visualization showing patterns across the week
    (day-of-week on x-axis) and day (hour on y-axis) with color intensity representing
    the data values.

    The returned Altair Chart object can be:
    - Displayed directly in Jupyter notebooks
    - Saved to HTML file using chart.save('filename.html')
    - Converted to HTML string using chart.to_html()
    - Faceted using chart.facet() for comparing multiple groups

    Args:
        calendar_data: Calendar data in wide or long format:
                      - Wide format: Series, numpy array, list, or DataFrame with 168 columns
                        (7 days × 24 hours). Multi-row DataFrames are automatically converted
                        to long format with the index used as the group column.
                      - Long format: DataFrame (pandas or Polars) with columns 'day_of_week',
                        'hour', 'value' (or 'num_events'), and optionally a group column for faceting.
                        Polars DataFrames are supported directly via narwhals.
        title: Chart title. If None, no title is shown.
        width: Chart width in pixels. Default is 400.
        height: Chart height in pixels. Default is 300.
        color_scheme: Altair color scheme name. Options include 'greens', 'blues',
                     'reds', 'oranges', 'purples', 'viridis', 'plasma', 'inferno',
                     'magma', 'cividis', etc. Default is 'greens'.
        monday_start: Start week on Monday (True) or Sunday (False). Default is True.
                     Only applies when converting from wide format.
        interactive: Enable tooltips and zoom/pan interactions. Default is True.
        show_values: Show values in tooltips when interactive=True. Default is True.

    Returns:
        Altair Chart object that can be displayed, saved, converted to HTML, or faceted

    Examples:
        Create and save to file:

        ```python
        import pandas as pd
        import numpy as np
        from latent_calendar.html import create_calendar_chart

        calendar_data = pd.Series(np.random.rand(168))
        chart = create_calendar_chart(calendar_data, title="Weekly Pattern")
        chart.save('calendar.html')
        ```

        Create faceted chart directly from cal.aggregate_events output:

        ```python
        from latent_calendar.datasets import load_chicago_bikes
        from latent_calendar.html import create_calendar_chart

        # Load and aggregate data by group
        df = load_chicago_bikes()
        df_agg = df.cal.aggregate_events("member_casual", "started_at")

        # Works directly - no need for dataframe_to_long_format!
        chart = create_calendar_chart(df_agg, width=250, height=200)

        # Apply faceting using Altair's .facet() method
        # The group column name comes from the DataFrame index name
        faceted = chart.facet(column='member_casual:N')
        faceted.save('faceted_calendar.html')
        ```

        Chain methods for compact code:

        ```python
        faceted = (
            create_calendar_chart(df_agg, color_scheme='viridis')
            .properties(width=250, height=200, title="Bike Share Patterns")
            .facet(column='member_casual:N', columns=2)
        )
        ```

        Customize appearance:

        ```python
        chart = create_calendar_chart(
            calendar_data,
            title="Custom Calendar",
            width=600,
            height=400,
            color_scheme='reds',
            monday_start=False  # Start on Sunday
        )
        ```

        Display in Jupyter notebooks:

        ```python
        # Chart will display automatically in Jupyter
        chart = create_calendar_chart(calendar_data)
        chart  # Display in notebook
        ```
    """
    # Handle dataframes (pandas or Polars) using narwhals for uniform column detection
    df_long = None
    try:
        df_nw = nw.from_native(calendar_data, eager_only=True)

        # Check if it's already in long format
        required_cols = {"day_of_week", "hour"}
        has_value = "value" in df_nw.columns
        has_num_events = "num_events" in df_nw.columns

        if required_cols.issubset(set(df_nw.columns)) and (has_value or has_num_events):
            # Already long format - rename num_events to value if needed, keep native type
            if has_num_events and not has_value:
                df_long = df_nw.rename({"num_events": "value"}).to_native()
            else:
                df_long = df_nw.to_native()
        elif df_nw.shape[1] == 168:
            # Wide format (168 columns) - convert via pandas
            df_pandas = df_nw.to_pandas()
            if len(df_pandas) == 1:
                df_long = wide_to_long_format(
                    df_pandas.iloc[0], monday_start=monday_start
                )
            else:
                group_col = df_pandas.index.name or "group"
                df_long = dataframe_to_long_format(
                    df_pandas, group_col=group_col, monday_start=monday_start
                )
        else:
            raise ValueError(
                f"DataFrame must have either (day_of_week, hour, value/num_events) columns "
                f"for long format, or 168 columns for wide format. "
                f"Got {df_nw.shape[1]} columns with columns: {df_nw.columns}"
            )
    except (TypeError, AttributeError):
        # Not a dataframe that narwhals can handle - try pandas fallback or array/list
        if isinstance(calendar_data, pd.DataFrame):
            required_cols = {"day_of_week", "hour", "value"}
            if required_cols.issubset(calendar_data.columns):
                df_long = calendar_data
            elif calendar_data.shape[1] == 168:
                if len(calendar_data) == 1:
                    df_long = wide_to_long_format(
                        calendar_data.iloc[0], monday_start=monday_start
                    )
                else:
                    group_col = calendar_data.index.name or "group"
                    df_long = dataframe_to_long_format(
                        calendar_data, group_col=group_col, monday_start=monday_start
                    )
            else:
                raise ValueError(
                    f"DataFrame must have either (day_of_week, hour, value) columns "
                    f"for long format, or 168 columns for wide format. "
                    f"Got {calendar_data.shape[1]} columns."
                )
        else:
            # Series, array, or list - convert to long format
            df_long = wide_to_long_format(calendar_data, monday_start=monday_start)

    # Day labels
    if monday_start:
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    else:
        day_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

    # Build tooltip configuration
    tooltip_list = []
    if show_values and interactive:
        tooltip_list = [
            alt.Tooltip("day_of_week:O", title="Day"),
            alt.Tooltip("hour:O", title="Hour"),
            alt.Tooltip("value:Q", title="Value", format=".2f"),
        ]

    # Create label expression for day names
    label_expr = f"datum.value === 0 ? '{day_labels[0]}' : datum.value === 1 ? '{day_labels[1]}' : datum.value === 2 ? '{day_labels[2]}' : datum.value === 3 ? '{day_labels[3]}' : datum.value === 4 ? '{day_labels[4]}' : datum.value === 5 ? '{day_labels[5]}' : '{day_labels[6]}'"

    # Build chart
    chart = (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "day_of_week:O",
                axis=alt.Axis(labelExpr=label_expr, title="Day of Week", labelAngle=0),
                scale=alt.Scale(domain=list(range(7))),
            ),
            y=alt.Y(
                "hour:O",
                axis=alt.Axis(title="Hour of Day"),
                scale=alt.Scale(domain=list(range(24))),
            ),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme=color_scheme),
                legend=alt.Legend(title="Count"),
            ),
            tooltip=tooltip_list,
        )
    )

    # Set properties, only add title if provided
    if title is not None:
        chart = chart.properties(width=width, height=height, title=title)
    else:
        chart = chart.properties(width=width, height=height)

    # Add interactivity if requested
    if interactive:
        chart = chart.interactive()

    return chart


def dataframe_to_long_format(
    df_wide: pd.DataFrame,
    group_col: str = "group",
    monday_start: bool = True,
) -> pd.DataFrame:
    """Convert wide calendar DataFrame with multiple rows to long format for Altair faceting.

    Transforms a DataFrame where each row represents a different group's calendar data
    (7 days × 24 hours = 168 columns) into a long-format DataFrame suitable for
    faceted Altair visualizations.

    Args:
        df_wide: DataFrame where each row is a calendar (168 columns) and index contains group names.
                Can be the result of df.cal.aggregate_events("group_col", "timestamp_col").
        group_col: Name for the column that will contain the group identifiers from the index.
                  Default is "group".
        monday_start: Start week on Monday (True) or Sunday (False). Default is True.

    Returns:
        DataFrame with columns: group, day_of_week (int 0-6), hour (int 0-23), value (float)

    Raises:
        ValueError: If DataFrame columns don't have exactly 168 values per row

    Examples:
        Convert multi-row aggregated data:

        ```python
        from latent_calendar.datasets import load_chicago_bikes
        from latent_calendar.html import dataframe_to_long_format

        df = load_chicago_bikes()
        df_agg = df.cal.aggregate_events("member_casual", "started_at")

        # Convert to long format for faceting
        df_long = dataframe_to_long_format(df_agg, group_col="rider_type")
        print(df_long.head())
        #    rider_type  day_of_week  hour  value
        # 0      member            0     0   42.0
        # 1      member            0     1   28.0
        # 2      member            0     2   15.0
        # 3      member            0     3    8.0
        # 4      member            0     4   12.0
        ```
    """
    # Get the number of columns (should be 168)
    n_cols = df_wide.shape[1]

    if n_cols != 168:
        raise ValueError(
            f"Expected 168 columns (7 days × 24 hours), got {n_cols}. "
            f"Make sure each row in df_wide is in calendar format with 168 time slots."
        )

    # Convert each row to long format and combine
    long_dfs = []
    for idx, row in df_wide.iterrows():
        # Convert single row to long format
        df_row_long = wide_to_long_format(row.values, monday_start=monday_start)
        # Add group identifier
        df_row_long[group_col] = str(idx)
        long_dfs.append(df_row_long)

    # Combine all rows
    df_long = pd.concat(long_dfs, ignore_index=True)

    # Reorder columns: group, day_of_week, hour, value
    df_long = df_long.loc[:, [group_col, "day_of_week", "hour", "value"]]

    return df_long
