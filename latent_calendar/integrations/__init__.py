"""Third-party integrations for latent-calendar.

This package provides integrations with popular visualization and mapping libraries
to make it easy to use latent-calendar with external tools.

Available integrations:
- folium: Embed calendar heatmaps in Folium maps (requires folium)

Examples:
    Using Folium integration:

    >>> import folium
    >>> from latent_calendar.integrations.folium import create_calendar_popup
    >>>
    >>> # Assuming calendar_data is your aggregated weekly data
    >>> popup = create_calendar_popup(
    ...     calendar_data,
    ...     title="Weekly Pattern"
    ... )
    >>>
    >>> # Add to Folium map
    >>> map = folium.Map(location=[37.8, -96], zoom_start=4)
    >>> folium.Marker([37.8, -96], popup=popup).add_to(map)
"""
