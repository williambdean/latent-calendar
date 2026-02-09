---
comments: true
---

# Folium Integration Module

The `integrations.folium` module provides functions for creating interactive map popups and tooltips with calendar heatmaps.

## Installation

```bash
pip install latent-calendar[html,folium]
```

This installs:
- `altair>=5.0.0` - For interactive charts
- `folium>=0.14.0` - For map creation

## Overview

This module bridges `latent-calendar` with Folium to display weekly temporal patterns on interactive maps. It's perfect for:

- Geographic time-series visualization
- Location-based event analysis
- Operational pattern mapping
- Research presentation with spatial context

See the [Folium Integration Guide](../../examples/folium-integration.md) for detailed examples and use cases.

## API Reference

::: latent_calendar.integrations.folium

## Quick Reference

### Main Functions

| Function | Returns | Use Case |
|----------|---------|----------|
| `create_calendar_popup()` | `folium.Popup` | Interactive popup on click |
| `create_calendar_tooltip()` | `folium.Tooltip` | Compact preview on hover |
| `create_popup_html()` | `str` | Advanced HTML customization |
| `create_tooltip_html()` | `str` | Advanced HTML customization |

### Common Parameters

All functions share these key parameters:

- `calendar_data`: pd.Series with 168 values (7 days × 24 hours)
- `title`: Optional title text
- `width`: Chart width in pixels
- `height`: Chart height in pixels
- `color_scheme`: Altair color scheme name
- `show_values`: Display numeric values in cells
- `monday_start`: Start week on Monday (vs Sunday)
- `interactive`: Enable zoom/pan
- `max_width`: Maximum popup/tooltip width

## Usage Examples

### Basic Popup

```python
import folium
from latent_calendar.datasets import load_ufo_sightings
from latent_calendar.integrations.folium import create_calendar_popup

# Prepare data
df = load_ufo_sightings()
df_states = df[df['country'] == 'us'].cal.aggregate_events(
    by='state/province',
    timestamp_col='Date_time'
)

# Create popup
popup = create_calendar_popup(
    df_states.loc['ca'],
    title="California UFO Sightings",
    width=400,
    height=280,
    color_scheme='blues'
)

# Add to map
m = folium.Map(location=[36.7, -119.7], zoom_start=6)
folium.Marker([36.7, -119.7], popup=popup).add_to(m)
m.save('map.html')
```

### Popup + Tooltip

```python
from latent_calendar.integrations.folium import (
    create_calendar_popup,
    create_calendar_tooltip
)

# Full calendar on click
popup = create_calendar_popup(
    data,
    title="Click for Details",
    width=500,
    height=350
)

# Compact preview on hover
tooltip = create_calendar_tooltip(
    data,
    title="Hover Preview",
    width=300,
    height=200,
    show_values=False
)

folium.Marker(
    location=[lat, lon],
    popup=popup,
    tooltip=tooltip
).add_to(map)
```

### GeoJSON with Popups

```python
import folium
import requests
from latent_calendar.integrations.folium import create_calendar_popup

# Load GeoJSON
geojson_data = requests.get(geojson_url).json()

# Add popup to each feature
for feature in geojson_data['features']:
    location_id = feature['properties']['id']
    calendar_data = df_locations.loc[location_id]

    popup = create_calendar_popup(
        calendar_data,
        title=feature['properties']['name'],
        width=400,
        height=280
    )

    folium.GeoJson(
        feature,
        popup=popup,
        style_function=lambda x: {'fillColor': 'blue'}
    ).add_to(map)
```

### Multiple Locations

```python
# Loop through locations and add popups
locations = {
    'New York': (40.7, -74.0),
    'Los Angeles': (34.0, -118.2),
    'Chicago': (41.9, -87.6)
}

for city, coords in locations.items():
    city_data = df_cities.loc[city]

    popup = create_calendar_popup(
        city_data,
        title=f"{city} Activity",
        color_scheme='viridis'
    )

    folium.CircleMarker(
        location=coords,
        radius=10,
        popup=popup,
        fill=True
    ).add_to(map)
```

### Custom HTML (Advanced)

```python
from latent_calendar.integrations.folium import create_popup_html
import folium

# Generate HTML string
html = create_popup_html(
    data,
    title="Custom Calendar",
    width=600,
    height=400,
    color_scheme='plasma',
    show_values=True
)

# Manually create IFrame and Popup for full control
iframe = folium.IFrame(html, width=650, height=480)
popup = folium.Popup(iframe, max_width=650)

folium.Marker([lat, lon], popup=popup).add_to(map)
```

## Color Schemes

All [Altair/Vega color schemes](https://vega.github.io/vega/docs/schemes/) are supported:

**Sequential:**
```python
color_scheme='blues'    # Blue gradient
color_scheme='greens'   # Green gradient
color_scheme='reds'     # Red gradient
color_scheme='viridis'  # Perceptually uniform
```

**Diverging:**
```python
color_scheme='redblue'      # Red to blue
color_scheme='purpleorange' # Purple to orange
```

## Performance Tips

For maps with many popups:

```python
# Optimize for performance
popup = create_calendar_popup(
    data,
    width=350,           # Smaller dimensions
    height=250,
    show_values=False,   # Hide text labels
    interactive=True     # Keep tooltips/zoom
)
```

**Results:**
- 10 popups with defaults: ~500 KB total
- 50 popups optimized: ~750 KB total
- 100 popups optimized: ~1.5 MB total

## Customization Tips

### Responsive Popups

Adjust size based on display:
```python
# Mobile-friendly
popup = create_calendar_popup(data, width=300, height=200)

# Desktop
popup = create_calendar_popup(data, width=600, height=400)
```

### Sunday Week Start

For US-style calendars:
```python
popup = create_calendar_popup(data, monday_start=False)
```

### No Interactivity (Static Display)

```python
popup = create_calendar_popup(
    data,
    interactive=False,  # No zoom/pan
    show_values=True    # Static values visible
)
```

### Wider Popups

For larger displays:
```python
popup = create_calendar_popup(
    data,
    width=700,
    height=500,
    max_width=800  # Popup container width
)
```

## Technical Details

### IFrame Embedding

The module uses `folium.IFrame` to properly embed complete HTML documents:

```python
# Internal implementation
html = create_popup_html(...)
iframe_height = height + 80  # Add space for title/margins
iframe = folium.IFrame(html, width=max_width, height=iframe_height)
popup = folium.Popup(iframe, max_width=max_width)
```

This creates proper isolation between map and chart HTML.

### Data Requirements

**Input Format:**
- Must be a `pd.Series` with 168 values
- Index: MultiIndex with (day_of_week, hour) pairs
- Values: Numeric (event counts, percentages, etc.)

**Example:**
```python
>>> data
day_of_week  hour
0            0       45
             1       32
             ...
6            23      67
Name: location_id, Length: 168, dtype: int64
```

Get this format using:
```python
df_wide = df.cal.aggregate_events(
    by='location_column',
    timestamp_col='datetime_column'
)
```

### HTML Structure

Generated popups contain:
1. Standalone HTML document with `<!DOCTYPE html>`
2. Embedded Vega-Lite specification (JSON)
3. Bundled JavaScript libraries (Vega, Vega-Lite, Vega-Embed)
4. Base64-encoded in iframe `src` attribute

### Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Mobile browsers**: Supported (may need smaller dimensions)

## Common Patterns

### Choropleth + Popups

```python
# Add base choropleth layer
folium.Choropleth(
    geo_data=geojson,
    data=totals,
    key_on='feature.id',
    fill_color='YlOrRd'
).add_to(map)

# Add calendar popups on top
for feature in geojson['features']:
    popup = create_calendar_popup(...)
    folium.GeoJson(
        feature,
        style_function=lambda x: {'fillOpacity': 0},
        popup=popup
    ).add_to(map)
```

### Marker Clusters with Popups

```python
from folium.plugins import MarkerCluster

marker_cluster = MarkerCluster().add_to(map)

for location in locations:
    popup = create_calendar_popup(...)
    folium.Marker(
        location,
        popup=popup
    ).add_to(marker_cluster)
```

## Troubleshooting

### Popup Not Displaying

**Symptom:** Click on marker, nothing happens
**Cause:** Missing `.add_to(map)` call
**Solution:** Always add markers/features to map:
```python
folium.Marker([lat, lon], popup=popup).add_to(map)
```

### Raw JavaScript in Popup

**Symptom:** Popup shows JavaScript code instead of chart
**Cause:** Old version without IFrame support
**Solution:** Update to latest version:
```bash
pip install --upgrade latent-calendar[html,folium]
```

### Chart Too Small/Large

**Symptom:** Chart doesn't fit in popup
**Cause:** Width/height mismatch with max_width
**Solution:** Ensure `max_width >= width`:
```python
popup = create_calendar_popup(
    data,
    width=500,
    max_width=600  # Must be >= width
)
```

### Slow Map Loading

**Symptom:** Map takes long time to load
**Cause:** Too many large popups
**Solution:** Optimize popup size:
```python
popup = create_calendar_popup(
    data,
    width=350,
    height=250,
    show_values=False
)
```

## See Also

- [Folium Integration Guide](../../examples/folium-integration.md) - Complete examples and use cases
- [HTML Module](../html.md) - Low-level HTML generation
- [`.cal` Accessor](../extensions.md) - Data preparation
- [UFO Sightings Example](../../examples/datasets/ufo-sightings.md) - Example dataset
- [Folium Documentation](https://python-visualization.github.io/folium/) - Official Folium docs
