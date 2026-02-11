"""Folium integration for latent-calendar visualizations.

This module provides helpers to embed interactive calendar heatmaps
into Folium map popups and tooltips. Calendar heatmaps show weekly patterns
across day-of-week and hour-of-day dimensions.

Note: This module requires folium to be installed:
    pip install latent-calendar[folium]

The module provides both raw HTML generation functions and convenience functions
that return ready-to-use Folium objects.

Examples:
    Basic usage with Folium markers:

    >>> import folium
    >>> from latent_calendar.integrations.folium import create_calendar_popup
    >>>
    >>> # Assuming df_states has aggregated calendar data (168 columns per state)
    >>> calendar_data = df_states.loc['CA']
    >>>
    >>> # Create popup with calendar heatmap
    >>> popup = create_calendar_popup(
    ...     calendar_data,
    ...     title="California Weekly Pattern",
    ...     color_scheme='blues'
    ... )
    >>>
    >>> # Add to map
    >>> map = folium.Map(location=[36.7, -119.7], zoom_start=6)
    >>> folium.Marker([36.7, -119.7], popup=popup).add_to(map)
    >>> map.save('map.html')

    Using with GeoJSON features:

    >>> import folium
    >>> from folium import GeoJson
    >>> from latent_calendar.integrations.folium import create_calendar_popup
    >>>
    >>> for feature in geojson_features:
    ...     state_name = feature['properties']['name']
    ...     calendar_data = df_states.loc[state_name]
    ...
    ...     popup = create_calendar_popup(
    ...         calendar_data,
    ...         title=f"Weekly Pattern: {state_name}",
    ...         width=450,
    ...         height=320
    ...     )
    ...
    ...     GeoJson(feature, popup=popup).add_to(map)

    Advanced: Get raw HTML for custom integration:

    >>> from latent_calendar.integrations.folium import create_popup_html
    >>>
    >>> html = create_popup_html(
    ...     calendar_data,
    ...     title="Custom Calendar",
    ...     color_scheme='reds'
    ... )
    >>>
    >>> # Use HTML string in your own way
    >>> custom_popup = folium.Popup(html, max_width=600)
"""

try:
    import folium
except ImportError as e:
    raise ImportError(
        "Folium integration requires folium to be installed. "
        "Install it directly or with: pip install latent-calendar[folium]"
    ) from e

from latent_calendar.html import create_calendar_chart


def create_popup_html(
    calendar_data,
    *,
    title: str | None = None,
    width: int = 400,
    height: int = 300,
    **chart_kwargs,
) -> str:
    """Create Folium-compatible popup HTML with calendar heatmap.

    Returns raw HTML string that can be used with folium.IFrame.
    This is useful for advanced users who want full control over the popup creation.

    For most use cases, consider using create_calendar_popup() which returns
    a ready-to-use folium.Popup object.

    Args:
        calendar_data: Wide calendar format (168 time slots: 7 days × 24 hours).
                      Can be pandas Series, numpy array, or list.
        title: Popup title displayed above the calendar. Optional.
        width: Calendar chart width in pixels. Default is 400.
        height: Calendar chart height in pixels. Default is 300.
        **chart_kwargs: Additional arguments passed to create_calendar_chart().
                       Useful options include:
                       - color_scheme: 'blues', 'greens', 'reds', 'viridis', etc.
                       - monday_start: True (default) or False for Sunday start
                       - interactive: True (default) for tooltips and zoom
                       - show_values: True (default) to show values in tooltips

    Returns:
        HTML string ready for folium.IFrame embedding

    Examples:
        Basic usage:

        >>> from latent_calendar.integrations.folium import create_popup_html
        >>> import folium
        >>>
        >>> html = create_popup_html(calendar_data, title="My Calendar")
        >>> iframe = folium.IFrame(html, width=450, height=350)
        >>> popup = folium.Popup(iframe, max_width=450)
        >>> folium.Marker([36.7, -119.7], popup=popup).add_to(map)

        With custom styling:

        >>> html = create_popup_html(
        ...     calendar_data,
        ...     title="Custom Calendar",
        ...     width=500,
        ...     height=350,
        ...     color_scheme='reds',
        ...     monday_start=False
        ... )
    """
    chart = create_calendar_chart(
        calendar_data,
        title=title,
        width=width,
        height=height,
        **chart_kwargs,
    )
    html = chart.to_html(embed_options={"actions": False})

    return html


def create_calendar_popup(
    calendar_data,
    *,
    title: str | None = None,
    width: int = 400,
    height: int = 300,
    max_width: int = 450,
    **chart_kwargs,
):
    """Create a Folium Popup object with embedded calendar heatmap.

    Convenience function that returns a ready-to-use folium.Popup object containing
    an interactive calendar visualization. This is the recommended way to create
    calendar popups for most use cases.

    The calendar shows weekly patterns with day-of-week on the x-axis and hour-of-day
    on the y-axis, with color intensity representing data values.

    Args:
        calendar_data: Wide calendar format (168 time slots: 7 days × 24 hours).
                      Can be pandas Series, numpy array, or list.
        title: Popup title displayed above the calendar. Optional.
        width: Calendar chart width in pixels. Default is 400.
        height: Calendar chart height in pixels. Default is 300.
        max_width: Maximum popup width in pixels. The popup will scale responsively
                  within this constraint. Default is 450.
        **chart_kwargs: Additional arguments passed to create_calendar_chart().
                       Useful options include:
                       - color_scheme: Color scheme name (default 'blues').
                         Options: 'blues', 'greens', 'reds', 'oranges', 'purples',
                         'viridis', 'plasma', 'inferno', 'magma', etc.
                       - monday_start: Start week on Monday (True, default) or Sunday (False)
                       - interactive: Enable tooltips and zoom (True, default)
                       - show_values: Show values in tooltips (True, default)

    Returns:
        folium.Popup object ready to add to markers or GeoJSON features

    Examples:
        Basic usage with markers:

        >>> import folium
        >>> from latent_calendar.integrations.folium import create_calendar_popup
        >>>
        >>> popup = create_calendar_popup(
        ...     calendar_data,
        ...     title="California UFO Sightings",
        ...     color_scheme='blues'
        ... )
        >>>
        >>> map = folium.Map(location=[36.7, -119.7], zoom_start=6)
        >>> folium.Marker([36.7, -119.7], popup=popup).add_to(map)
        >>> map.save('map.html')

        With GeoJSON features:

        >>> from folium import GeoJson
        >>> from latent_calendar.integrations.folium import create_calendar_popup
        >>>
        >>> for feature in geojson_data['features']:
        ...     state_name = feature['properties']['name']
        ...     calendar_data = df_states.loc[state_name]
        ...
        ...     popup = create_calendar_popup(
        ...         calendar_data,
        ...         title=f"Patterns: {state_name}",
        ...         width=450,
        ...         height=320,
        ...         max_width=500
        ...     )
        ...
        ...     GeoJson(feature, popup=popup).add_to(map)

        Custom styling:

        >>> popup = create_calendar_popup(
        ...     calendar_data,
        ...     title="Weekend Activity",
        ...     width=500,
        ...     height=350,
        ...     color_scheme='viridis',
        ...     monday_start=False  # Start week on Sunday
        ... )
    """
    html = create_popup_html(
        calendar_data,
        title=title,
        width=width,
        height=height,
        **chart_kwargs,
    )

    # Use IFrame to properly embed the full HTML document
    # IFrame height needs to account for chart + title + padding
    iframe_height = height + 80  # Add space for title and margins
    iframe = folium.IFrame(
        html,
        width=max_width,
        height=iframe_height,
    )

    return folium.Popup(iframe, max_width=max_width)


def create_tooltip_html(
    calendar_data,
    *,
    title: str | None = None,
    width: int = 300,
    height: int = 200,
    compact: bool = True,
    **chart_kwargs,
) -> str:
    """Create Folium-compatible tooltip HTML with calendar heatmap.

    Returns raw HTML string for a smaller, optimized calendar suitable for hover
    tooltips (as opposed to click popups). Tooltips should be more compact since
    they appear on hover and should not obstruct the map view.

    Args:
        calendar_data: Wide calendar format (168 time slots: 7 days × 24 hours)
        title: Tooltip title. Optional.
        width: Tooltip width in pixels (default 300, smaller than popup default)
        height: Tooltip height in pixels (default 200, smaller than popup default)
        compact: Use compact styling for smaller display. When True, disables
                some interactive features to reduce size. Default is True.
        **chart_kwargs: Additional arguments passed to create_calendar_chart()

    Returns:
        HTML string ready for folium.Tooltip()

    Note:
        Tooltips are shown on hover and should be smaller/simpler than popups.
        Consider using compact=True (default) to disable some features and
        reduce the size.

    Examples:
        >>> from latent_calendar.integrations.folium import create_tooltip_html
        >>> import folium
        >>>
        >>> html = create_tooltip_html(
        ...     calendar_data,
        ...     title="Hover Preview",
        ...     compact=True
        ... )
        >>> tooltip = folium.Tooltip(html)
    """
    if compact:
        # Disable some features for compact display
        chart_kwargs.setdefault("interactive", False)
        chart_kwargs.setdefault("show_values", False)

    return create_popup_html(
        calendar_data,
        title=title,
        width=width,
        height=height,
        **chart_kwargs,
    )


def create_calendar_tooltip(
    calendar_data,
    *,
    title: str | None = None,
    width: int = 300,
    height: int = 200,
    compact: bool = True,
    **chart_kwargs,
) -> folium.Tooltip:
    """Create a Folium Tooltip object with embedded calendar heatmap.

    Convenience function that returns a ready-to-use folium.Tooltip object containing
    a compact calendar visualization. Tooltips appear on hover and should be smaller
    and simpler than popups.

    Args:
        calendar_data: Wide calendar format (168 time slots: 7 days × 24 hours)
        title: Tooltip title. Optional.
        width: Calendar chart width in pixels (default 300, smaller than popups)
        height: Calendar chart height in pixels (default 200, smaller than popups)
        compact: Use compact styling that disables some interactive features to
                reduce size and improve hover performance. Default is True.
        **chart_kwargs: Additional arguments passed to create_calendar_chart()

    Returns:
        folium.Tooltip object ready to add to markers or features

    Examples:
        >>> import folium
        >>> from latent_calendar.integrations.folium import create_calendar_tooltip
        >>>
        >>> tooltip = create_calendar_tooltip(
        ...     calendar_data,
        ...     title="Hover Preview",
        ...     compact=True
        ... )
        >>>
        >>> folium.Marker(
        ...     [36.7, -119.7],
        ...     tooltip=tooltip
        ... ).add_to(map)
    """
    html = create_tooltip_html(
        calendar_data,
        title=title,
        width=width,
        height=height,
        compact=compact,
        **chart_kwargs,
    )

    return folium.Tooltip(html)
