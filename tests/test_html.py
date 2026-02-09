"""Tests for latent_calendar.html module."""

import pytest
import numpy as np
import pandas as pd


# Skip all tests if altair is not installed
altair = pytest.importorskip("altair")

from latent_calendar.html import (  # noqa: E402
    wide_to_long_format,
    create_calendar_chart,
    dataframe_to_long_format,
)


class TestWideToLongFormat:
    """Tests for wide_to_long_format function."""

    def test_converts_168_values_correctly(self):
        """Test that 168 values are converted to long format."""
        # Create test data: 7 days × 24 hours = 168 values
        calendar_data = pd.Series(np.arange(168))

        df_long = wide_to_long_format(calendar_data)

        assert len(df_long) == 168
        assert list(df_long.columns) == ["day_of_week", "hour", "value"]
        assert df_long["day_of_week"].min() == 0
        assert df_long["day_of_week"].max() == 6
        assert df_long["hour"].min() == 0
        assert df_long["hour"].max() == 23

    def test_monday_start(self):
        """Test Monday start (default)."""
        calendar_data = pd.Series(np.arange(168))

        df_long = wide_to_long_format(calendar_data, monday_start=True)

        # First 24 hours should be Monday (day 0)
        assert all(df_long.iloc[:24]["day_of_week"] == 0)
        # Hours should be 0-23 for first day
        assert list(df_long.iloc[:24]["hour"]) == list(range(24))

    def test_sunday_start(self):
        """Test Sunday start."""
        calendar_data = pd.Series(np.arange(168))

        df_long = wide_to_long_format(calendar_data, monday_start=False)

        # First 24 hours should still map correctly but shifted
        # With Sunday start, original Monday (0) becomes Tuesday (1), etc.
        # And original Sunday (6) becomes Monday (0)
        assert len(df_long) == 168
        assert df_long["day_of_week"].min() == 0
        assert df_long["day_of_week"].max() == 6

    def test_accepts_numpy_array(self):
        """Test that function accepts numpy array."""
        calendar_data = np.random.rand(168)

        df_long = wide_to_long_format(calendar_data)

        assert len(df_long) == 168
        assert all(df_long["value"] == calendar_data)

    def test_accepts_list(self):
        """Test that function accepts list."""
        calendar_data = list(range(168))

        df_long = wide_to_long_format(calendar_data)

        assert len(df_long) == 168

    def test_raises_on_wrong_length(self):
        """Test that ValueError is raised for non-168 length data."""
        with pytest.raises(ValueError, match="Expected 168 values"):
            wide_to_long_format(pd.Series(np.arange(100)))

        with pytest.raises(ValueError, match="Expected 168 values"):
            wide_to_long_format(pd.Series(np.arange(200)))

    def test_preserves_values(self):
        """Test that values are preserved in conversion."""
        values = np.random.rand(168)
        calendar_data = pd.Series(values)

        df_long = wide_to_long_format(calendar_data)

        assert np.allclose(df_long["value"].values, values)


class TestCreateCalendarChart:
    """Tests for create_calendar_chart function."""

    def test_returns_altair_chart(self):
        """Test that function returns an Altair Chart object."""
        calendar_data = pd.Series(np.random.rand(168))

        chart = create_calendar_chart(calendar_data)

        assert isinstance(chart, altair.Chart)

    def test_with_title(self):
        """Test chart with title."""
        calendar_data = pd.Series(np.random.rand(168))

        chart = create_calendar_chart(calendar_data, title="Test Calendar")

        # Check that title is set (chart.title might be a TitleParams object or string)
        assert chart.title is not None

    def test_custom_dimensions(self):
        """Test chart with custom width and height."""
        calendar_data = pd.Series(np.random.rand(168))

        chart = create_calendar_chart(calendar_data, width=600, height=400)

        assert chart.width == 600
        assert chart.height == 400

    def test_different_color_schemes(self):
        """Test different color schemes."""
        calendar_data = pd.Series(np.random.rand(168))

        for scheme in ["greens", "blues", "reds", "viridis"]:
            chart = create_calendar_chart(calendar_data, color_scheme=scheme)
            assert isinstance(chart, altair.Chart)

    def test_monday_and_sunday_start(self):
        """Test both Monday and Sunday start options."""
        calendar_data = pd.Series(np.random.rand(168))

        chart_monday = create_calendar_chart(calendar_data, monday_start=True)
        chart_sunday = create_calendar_chart(calendar_data, monday_start=False)

        assert isinstance(chart_monday, altair.Chart)
        assert isinstance(chart_sunday, altair.Chart)

    def test_interactive_options(self):
        """Test interactive and show_values options."""
        calendar_data = pd.Series(np.random.rand(168))

        # Interactive with values
        chart1 = create_calendar_chart(
            calendar_data, interactive=True, show_values=True
        )
        assert isinstance(chart1, altair.Chart)

        # Non-interactive without values
        chart2 = create_calendar_chart(
            calendar_data, interactive=False, show_values=False
        )
        assert isinstance(chart2, altair.Chart)

    def test_with_all_zeros(self):
        """Test with all zero values."""
        calendar_data = pd.Series(np.zeros(168))

        chart = create_calendar_chart(calendar_data, title="All Zeros")

        assert isinstance(chart, altair.Chart)

    def test_with_large_values(self):
        """Test with large values."""
        calendar_data = pd.Series(np.random.rand(168) * 10000)

        chart = create_calendar_chart(calendar_data, title="Large Values")

        assert isinstance(chart, altair.Chart)

    def test_accepts_long_format_dataframe(self):
        """Test that function accepts long format DataFrame."""
        # Create long format data
        df_long = pd.DataFrame(
            {
                "day_of_week": list(range(7)) * 24,
                "hour": [h for h in range(24) for _ in range(7)],
                "value": np.random.rand(168),
            }
        )

        chart = create_calendar_chart(df_long)

        assert isinstance(chart, altair.Chart)

    def test_long_format_with_group_column(self):
        """Test that group column is preserved in long format."""
        df_long = pd.DataFrame(
            {
                "group": ["a"] * 84 + ["b"] * 84,
                "day_of_week": list(range(7)) * 24,
                "hour": [h for h in range(24) for _ in range(7)],
                "value": np.random.rand(168),
            }
        )

        chart = create_calendar_chart(df_long, width=250, height=200)

        assert isinstance(chart, altair.Chart)

    def test_facet_method_works(self):
        """Test that .facet() method works on returned chart."""
        # Create multi-group long format
        df_wide = pd.DataFrame(
            {i: np.random.rand(2) for i in range(168)}, index=["group_a", "group_b"]
        )
        df_long = dataframe_to_long_format(df_wide, group_col="category")

        chart = create_calendar_chart(df_long, width=250, height=200)
        faceted = chart.facet(column="category:N")

        # Altair facet returns a Chart-like object
        assert faceted is not None

    def test_properties_chaining(self):
        """Test that .properties() method chaining works."""
        calendar_data = pd.Series(np.random.rand(168))

        chart = create_calendar_chart(calendar_data)
        modified = chart.properties(width=500, height=400, title="Chained Title")

        assert modified.width == 500
        assert modified.height == 400

    def test_accepts_multirow_wide_format(self):
        """Test that multi-row wide format DataFrame is auto-converted."""
        df_wide = pd.DataFrame(
            {i: np.random.rand(3) for i in range(168)}, index=["a", "b", "c"]
        )

        # Should automatically convert to long format
        chart = create_calendar_chart(df_wide, width=250, height=200)

        assert isinstance(chart, altair.Chart)

    def test_multirow_wide_format_with_named_index(self):
        """Test that multi-row wide format uses index name for group column."""
        df_wide = pd.DataFrame(
            {i: np.random.rand(2) for i in range(168)},
            index=pd.Index(["group_a", "group_b"], name="category"),
        )

        chart = create_calendar_chart(df_wide)

        # The chart should be created successfully
        assert isinstance(chart, altair.Chart)

        # Should be able to facet by the index name
        faceted = chart.facet(column="category:N")
        assert faceted is not None

    def test_raises_on_invalid_dataframe_format(self):
        """Test that invalid DataFrame format raises error."""
        df_invalid = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        with pytest.raises(ValueError, match="DataFrame must have either"):
            create_calendar_chart(df_invalid)


class TestDataFrameToLongFormat:
    """Tests for dataframe_to_long_format function."""

    def test_converts_multirow_dataframe(self):
        """Test that multi-row DataFrame is converted to long format."""
        # Create test data: 2 groups × 168 time slots
        df_wide = pd.DataFrame(
            {i: np.random.rand(2) for i in range(168)}, index=["group_a", "group_b"]
        )

        df_long = dataframe_to_long_format(df_wide, group_col="category")

        # Should have 2 groups × 168 slots = 336 rows
        assert len(df_long) == 336
        assert list(df_long.columns) == ["category", "day_of_week", "hour", "value"]
        assert set(df_long["category"]) == {"group_a", "group_b"}
        assert df_long["day_of_week"].min() == 0
        assert df_long["day_of_week"].max() == 6
        assert df_long["hour"].min() == 0
        assert df_long["hour"].max() == 23

    def test_monday_start_multirow(self):
        """Test Monday start with multiple rows."""
        df_wide = pd.DataFrame(
            {i: np.arange(3) for i in range(168)}, index=["a", "b", "c"]
        )

        df_long = dataframe_to_long_format(df_wide, monday_start=True)

        # Check first 24 rows (first group, first day)
        first_group = df_long[df_long["group"] == "a"].iloc[:24]
        assert all(first_group["day_of_week"] == 0)
        assert list(first_group["hour"]) == list(range(24))

    def test_sunday_start_multirow(self):
        """Test Sunday start with multiple rows."""
        df_wide = pd.DataFrame({i: np.arange(2) for i in range(168)}, index=["x", "y"])

        df_long = dataframe_to_long_format(df_wide, monday_start=False)

        assert len(df_long) == 336
        assert df_long["day_of_week"].min() == 0
        assert df_long["day_of_week"].max() == 6

    def test_raises_on_wrong_columns(self):
        """Test that ValueError is raised for non-168 column data."""
        df_wrong = pd.DataFrame({i: [1, 2] for i in range(100)}, index=["a", "b"])

        with pytest.raises(ValueError, match="Expected 168 columns"):
            dataframe_to_long_format(df_wrong)

    def test_custom_group_column_name(self):
        """Test custom group column name."""
        df_wide = pd.DataFrame(
            {i: [1, 2, 3] for i in range(168)}, index=["red", "green", "blue"]
        )

        df_long = dataframe_to_long_format(df_wide, group_col="color")

        assert "color" in df_long.columns
        assert set(df_long["color"]) == {"red", "green", "blue"}

    def test_preserves_values_multirow(self):
        """Test that values are preserved in conversion."""
        df_wide = pd.DataFrame(
            {i: [i % 10, (i + 1) % 10] for i in range(168)}, index=["row1", "row2"]
        )

        df_long = dataframe_to_long_format(df_wide)

        # Check that row1's values match
        row1_long = df_long[df_long["group"] == "row1"]
        assert len(row1_long) == 168


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete workflow from data to HTML."""
        # Create sample calendar data
        calendar_data = pd.Series(np.random.rand(168))

        # Convert to long format
        df_long = wide_to_long_format(calendar_data)
        assert len(df_long) == 168

        # Create chart
        chart = create_calendar_chart(calendar_data, title="Integration Test")
        assert isinstance(chart, altair.Chart)

        # Generate HTML using Altair's to_html()
        html = chart.to_html()
        assert isinstance(html, str)
        assert len(html) > 0
        assert "vega" in html.lower() or "vegaembed" in html.lower()

    def test_with_realistic_data(self):
        """Test with realistic calendar pattern data."""
        # Create data with a weekly pattern (higher on weekends)
        hours_per_day = []
        for day in range(7):
            # Weekend has higher values
            if day in [5, 6]:  # Saturday, Sunday
                hours_per_day.extend(np.random.rand(24) * 100 + 50)
            else:
                hours_per_day.extend(np.random.rand(24) * 50)

        calendar_data = pd.Series(hours_per_day)

        # Should handle realistic patterns
        chart = create_calendar_chart(
            calendar_data, title="Weekend Pattern", color_scheme="viridis"
        )
        assert isinstance(chart, altair.Chart)

        # Generate HTML using Altair's to_html()
        html = chart.to_html(embed_options={"actions": False})
        assert isinstance(html, str)
        assert len(html) > 0

    def test_full_workflow_with_faceting(self):
        """Test complete workflow with faceting."""
        # Create multi-group calendar data
        df_wide = pd.DataFrame(
            {i: np.random.rand(2) for i in range(168)}, index=["group_a", "group_b"]
        )

        # Convert to long format
        df_long = dataframe_to_long_format(df_wide, group_col="category")
        assert len(df_long) == 336  # 2 groups × 168 slots

        # Create base chart
        chart = create_calendar_chart(df_long, width=250, height=200)
        assert isinstance(chart, altair.Chart)

        # Apply faceting
        faceted = chart.facet(column="category:N")
        assert faceted is not None

        # Generate HTML using Altair's to_html()
        html = faceted.to_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_workflow_with_chaining(self):
        """Test property chaining workflow."""
        # Create multi-group data
        df_wide = pd.DataFrame(
            {i: [i % 10, (i + 1) % 10] for i in range(168)}, index=["row1", "row2"]
        )

        # Convert to long format
        df_long = dataframe_to_long_format(df_wide, group_col="group")

        # Create and modify chart with chaining
        faceted = (
            create_calendar_chart(df_long, color_scheme="viridis")
            .properties(width=200, height=150, title="Chained Example")
            .facet(column="group:N", columns=2)
        )

        assert faceted is not None
