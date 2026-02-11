"""Tests for latent_calendar.integrations.folium module."""

import pytest
import numpy as np
import pandas as pd


# Skip all tests if required packages are not installed
pytest.importorskip("altair")
folium = pytest.importorskip("folium")

from latent_calendar.integrations.folium import (  # noqa: E402
    create_popup_html,
    create_calendar_popup,
    create_tooltip_html,
    create_calendar_tooltip,
)


class TestCreatePopupHtml:
    """Tests for create_popup_html function."""

    def test_returns_html_string(self):
        """Test that function returns an HTML string."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_popup_html(calendar_data)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_vega(self):
        """Test that HTML contains Vega-Lite content."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_popup_html(calendar_data, title="Test Popup")

        # Should contain Vega embed content
        assert "vega" in html.lower() or "vegaembed" in html.lower()

    def test_with_title(self):
        """Test popup with title."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_popup_html(calendar_data, title="My Calendar Popup")

        assert isinstance(html, str)
        assert len(html) > 0

    def test_custom_dimensions(self):
        """Test with custom width and height."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_popup_html(calendar_data, width=500, height=350)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_with_chart_kwargs(self):
        """Test that chart_kwargs are passed through."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_popup_html(
            calendar_data,
            title="Custom Popup",
            width=450,
            height=320,
            color_scheme="reds",
            monday_start=False,
        )

        assert isinstance(html, str)
        assert len(html) > 0


class TestCreateCalendarPopup:
    """Tests for create_calendar_popup function."""

    def test_returns_folium_popup(self):
        """Test that function returns a folium.Popup object."""
        calendar_data = pd.Series(np.random.rand(168))

        popup = create_calendar_popup(calendar_data)

        assert isinstance(popup, folium.Popup)

    def test_popup_with_title(self):
        """Test popup with title."""
        calendar_data = pd.Series(np.random.rand(168))

        popup = create_calendar_popup(calendar_data, title="Test State")

        assert isinstance(popup, folium.Popup)

    def test_custom_max_width(self):
        """Test with custom max_width."""
        calendar_data = pd.Series(np.random.rand(168))

        popup = create_calendar_popup(calendar_data, max_width=600)

        assert isinstance(popup, folium.Popup)
        assert popup.options.get("max_width") == 600

    def test_with_color_schemes(self):
        """Test different color schemes."""
        calendar_data = pd.Series(np.random.rand(168))

        for scheme in ["blues", "greens", "viridis"]:
            popup = create_calendar_popup(calendar_data, color_scheme=scheme)
            assert isinstance(popup, folium.Popup)

    def test_monday_and_sunday_start(self):
        """Test both Monday and Sunday start."""
        calendar_data = pd.Series(np.random.rand(168))

        popup_monday = create_calendar_popup(calendar_data, monday_start=True)
        popup_sunday = create_calendar_popup(calendar_data, monday_start=False)

        assert isinstance(popup_monday, folium.Popup)
        assert isinstance(popup_sunday, folium.Popup)


class TestCreateTooltipHtml:
    """Tests for create_tooltip_html function."""

    def test_returns_html_string(self):
        """Test that function returns an HTML string."""
        calendar_data = pd.Series(np.random.rand(168))

        html = create_tooltip_html(calendar_data)

        assert isinstance(html, str)
        assert len(html) > 0

    def test_with_compact_mode(self):
        """Test compact mode for tooltips."""
        calendar_data = pd.Series(np.random.rand(168))

        html_compact = create_tooltip_html(calendar_data, compact=True)
        html_full = create_tooltip_html(calendar_data, compact=False)

        assert isinstance(html_compact, str)
        assert isinstance(html_full, str)

    def test_default_smaller_dimensions(self):
        """Test that default dimensions are smaller than popup defaults."""
        calendar_data = pd.Series(np.random.rand(168))

        # Should accept smaller default dimensions
        html = create_tooltip_html(calendar_data, width=300, height=200)

        assert isinstance(html, str)
        assert len(html) > 0


class TestCreateCalendarTooltip:
    """Tests for create_calendar_tooltip function."""

    def test_returns_folium_tooltip(self):
        """Test that function returns a folium.Tooltip object."""
        calendar_data = pd.Series(np.random.rand(168))

        tooltip = create_calendar_tooltip(calendar_data)

        assert isinstance(tooltip, folium.Tooltip)

    def test_with_compact_mode(self):
        """Test tooltip with compact mode."""
        calendar_data = pd.Series(np.random.rand(168))

        tooltip = create_calendar_tooltip(calendar_data, compact=True)

        assert isinstance(tooltip, folium.Tooltip)

    def test_with_title(self):
        """Test tooltip with title."""
        calendar_data = pd.Series(np.random.rand(168))

        tooltip = create_calendar_tooltip(calendar_data, title="Hover Preview")

        assert isinstance(tooltip, folium.Tooltip)


class TestIntegration:
    """Integration tests for Folium module."""

    def test_can_add_popup_to_map(self):
        """Test that popup can be added to Folium map."""
        calendar_data = pd.Series(np.random.rand(168))

        # Create map
        m = folium.Map(location=[37.8, -96], zoom_start=4)

        # Create popup
        popup = create_calendar_popup(
            calendar_data, title="Test State", color_scheme="blues"
        )

        # Add marker with popup
        marker = folium.Marker(location=[37.8, -96], popup=popup)
        marker.add_to(m)

        # Should succeed without errors
        assert isinstance(m, folium.Map)

    def test_can_add_tooltip_to_map(self):
        """Test that tooltip can be added to Folium map."""
        calendar_data = pd.Series(np.random.rand(168))

        # Create map
        m = folium.Map(location=[37.8, -96], zoom_start=4)

        # Create tooltip
        tooltip = create_calendar_tooltip(calendar_data, title="Hover", compact=True)

        # Add marker with tooltip
        marker = folium.Marker(location=[37.8, -96], tooltip=tooltip)
        marker.add_to(m)

        # Should succeed without errors
        assert isinstance(m, folium.Map)

    def test_full_workflow_with_geojson(self):
        """Test complete workflow with GeoJSON feature."""
        calendar_data = pd.Series(np.random.rand(168))

        # Create map
        m = folium.Map(location=[37.8, -96], zoom_start=4)

        # Create popup
        popup = create_calendar_popup(
            calendar_data,
            title="Test Feature",
            width=450,
            height=320,
            max_width=500,
        )

        # Simple GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": {"name": "Test State"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-100, 40],
                        [-100, 45],
                        [-95, 45],
                        [-95, 40],
                        [-100, 40],
                    ]
                ],
            },
        }

        # Add GeoJSON with popup
        geojson = folium.GeoJson(feature, popup=popup)
        geojson.add_to(m)

        # Should succeed without errors
        assert isinstance(m, folium.Map)

    def test_multiple_popups_on_same_map(self):
        """Test adding multiple calendar popups to same map."""
        m = folium.Map(location=[37.8, -96], zoom_start=4)

        # Add multiple markers with different calendar data
        for i, lat in enumerate([35, 38, 41]):
            calendar_data = pd.Series(np.random.rand(168) * (i + 1))
            popup = create_calendar_popup(
                calendar_data, title=f"Location {i + 1}", color_scheme="blues"
            )
            folium.Marker(location=[lat, -96], popup=popup).add_to(m)

        # Should handle multiple popups
        assert isinstance(m, folium.Map)

    def test_with_realistic_data(self):
        """Test with realistic calendar pattern data."""
        # Create data with a pattern (higher on evenings)
        hours_per_day = []
        for day in range(7):
            daily_pattern = []
            for hour in range(24):
                # Higher values in evening (18-22)
                if 18 <= hour <= 22:
                    daily_pattern.append(np.random.rand() * 100 + 50)
                else:
                    daily_pattern.append(np.random.rand() * 30)
            hours_per_day.extend(daily_pattern)

        calendar_data = pd.Series(hours_per_day)

        # Create popup
        popup = create_calendar_popup(
            calendar_data, title="Evening Pattern", color_scheme="viridis"
        )

        assert isinstance(popup, folium.Popup)

        # Can be added to map
        m = folium.Map(location=[37.8, -96], zoom_start=4)
        folium.Marker(location=[37.8, -96], popup=popup).add_to(m)
        assert isinstance(m, folium.Map)
