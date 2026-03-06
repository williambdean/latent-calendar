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

## Usage Example

### Quick Start

Here's a minimal example showing UFO sightings in California:

```python
import folium
from latent_calendar.datasets import load_ufo_sightings
from latent_calendar.integrations.folium import create_calendar_popup

# 1. Load and aggregate data by state
df = load_ufo_sightings()
df_states = df[df['country'] == 'us'].cal.aggregate_events(
    by='state/province',
    timestamp_col='Date_time'
)

# 2. Get California's weekly pattern
california_data = df_states.loc['ca']

# 3. Create an interactive popup
popup = create_calendar_popup(
    california_data,
    title="California UFO Sightings",
    width=400,
    height=280,
    color_scheme='blues'
)

# 4. Add to map
m = folium.Map(location=[36.7, -119.7], zoom_start=6)
folium.Marker(
    location=[36.7, -119.7],
    popup=popup,
    tooltip="Click to see calendar"
).add_to(m)

m.save('california_ufos.html')
```

For complete examples with GeoJSON, multiple popups, and visual demonstrations, see the [Folium Integration Guide](../../examples/folium-integration.md).

## Customization Options

### Color Schemes

All [Altair/Vega color schemes](https://vega.github.io/vega/docs/schemes/) are supported. Common choices:

```python
color_scheme='blues'    # Blue gradient (default for popups)
color_scheme='greens'   # Green gradient
color_scheme='viridis'  # Perceptually uniform
color_scheme='reds'     # Red gradient
```

### Week Start Day

```python
# Start week on Monday (default)
popup = create_calendar_popup(data, monday_start=True)

# Start week on Sunday (US convention)
popup = create_calendar_popup(data, monday_start=False)
```

### Interactive Features

```python
# Enable zoom/pan and tooltips (default)
popup = create_calendar_popup(data, interactive=True)

# Disable interactivity for static display
popup = create_calendar_popup(data, interactive=False, show_values=True)
```

### Chart Dimensions

```python
# Compact popup for mobile
popup = create_calendar_popup(data, width=300, height=200, max_width=350)

# Large popup for desktop
popup = create_calendar_popup(data, width=600, height=400, max_width=650)
```

## Data Requirements

The folium integration expects calendar data in **wide format** - a pandas Series with 168 values representing 7 days × 24 hours:

```python
# Get wide format using .cal.aggregate_events()
df_aggregated = df.cal.aggregate_events(
    by='location_column',
    timestamp_col='datetime_column'
)

# Each row is a location with 168 time slots
>>> df_aggregated.loc['ca']
day_of_week  hour
0            0       45
             1       32
             ...
6            23      67
Name: ca, dtype: int64
```

## Common Issues

**Popup not appearing?** Make sure you call `.add_to(map)`:
```python
folium.Marker([lat, lon], popup=popup).add_to(m)  # Don't forget this!
```

**Chart too small/large?** Ensure `max_width >= width`:
```python
popup = create_calendar_popup(data, width=500, max_width=600)
```

**Map loading slowly?** Use smaller dimensions and disable values:
```python
popup = create_calendar_popup(data, width=350, height=250, show_values=False)
```

## See Also

- [Folium Integration Guide](../../examples/folium-integration.md) - Complete examples and use cases
- [HTML Module](../html.md) - Low-level HTML generation
- [`.cal` Accessor](../extensions.md) - Data preparation
- [UFO Sightings Example](../../examples/datasets/ufo-sightings.md) - Example dataset
- [Folium Documentation](https://python-visualization.github.io/folium/) - Official Folium docs
