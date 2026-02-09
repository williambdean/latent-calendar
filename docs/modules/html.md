---
comments: true
---

# HTML Module

The `html` module provides functions for converting calendar data into interactive HTML visualizations using [Altair](https://altair-viz.github.io/).

## Installation

```bash
pip install latent-calendar[html]
```

This installs `altair>=5.0.0` as an optional dependency.

## Overview

The HTML module transforms weekly calendar data (168 time slots) into interactive heatmap charts that can be:

- Embedded in web pages
- Used in Jupyter notebooks
- Integrated with mapping libraries (see [Folium Integration](../examples/folium-integration.md))

## API Reference

::: latent_calendar.html

## Usage Examples

### Basic HTML Generation

```python
from latent_calendar.datasets import load_ufo_sightings
from latent_calendar.html import create_calendar_chart

# Load and aggregate data
df = load_ufo_sightings()
df_states = df[df['country'] == 'us'].cal.aggregate_events(
    by='state/province',
    timestamp_col='Date_time'
)

# Create chart for California
chart = create_calendar_chart(
    df_states.loc['ca'],
    title="California UFO Sightings",
    width=500,
    height=350,
    color_scheme='greens'
)

# Save directly to file
chart.save('calendar.html')

# Or get HTML string for embedding
html = chart.to_html()
with open('calendar.html', 'w') as f:
    f.write(html)
```

### Faceted Charts (Comparing Multiple Groups)

Create side-by-side calendar comparisons using Altair's `.facet()` method.

**Direct usage with `cal.aggregate_events()`:**

```python
from latent_calendar.datasets import load_chicago_bikes
from latent_calendar.html import create_calendar_chart

# Load data and aggregate by group
df = load_chicago_bikes()
df_agg = df.cal.aggregate_events("member_casual", "started_at")

# Create base chart directly from aggregated data
# The function automatically detects multi-row format and converts it
chart = create_calendar_chart(
    df_agg,
    width=250,
    height=200,
    color_scheme='viridis'
)

# Apply faceting using Altair's .facet() method
# Use the index name from aggregate_events as the grouping column
faceted = chart.facet(column='member_casual:N')
faceted.save('faceted_calendar.html')
```

**Chain methods for compact code:**

```python
# Create chart with custom properties and faceting in one expression
faceted = (
    create_calendar_chart(df_agg, color_scheme='viridis')
    .properties(width=250, height=200, title="Bike Share Patterns")
    .facet(column='member_casual:N', columns=2)
)
faceted.save('faceted_calendar.html')
```

**Using explicit conversion (optional):**

If you need more control over the group column name, you can still use `dataframe_to_long_format()`:

```python
from latent_calendar.html import dataframe_to_long_format

# Explicitly convert with custom group column name
df_long = dataframe_to_long_format(df_agg, group_col="rider_type")

chart = create_calendar_chart(df_long)
faceted = chart.facet(column='rider_type:N')
```

**Advanced faceting options:**

```python
# Two-dimensional faceting (e.g., by season and rider type)
faceted_2d = chart.facet(
    row='season:N',
    column='rider_type:N'
)

# Custom number of columns in facet grid
faceted = chart.facet(
    column='station_name:N',
    columns=3  # 3 charts per row
)

# Adjust sizing before faceting
smaller_chart = chart.properties(width=200, height=150)
faceted = smaller_chart.facet(column='group:N')
```

### Creating Altair Chart Objects

```python
from latent_calendar.html import create_calendar_chart

# Create a chart object for further customization
chart = create_calendar_chart(
    df_states.loc['ca'],
    title="Weekly Pattern",
    width=400,
    height=300,
    color_scheme='viridis',
    show_values=True,
    monday_start=True
)

# Customize further with Altair API
chart = chart.configure_view(strokeWidth=0)

# Export as HTML
html = chart.to_html()
```

### Jupyter Notebook Display

```python
from latent_calendar.html import create_calendar_chart

# Create and display chart in notebook
chart = create_calendar_chart(
    calendar_data,
    title="Activity Pattern",
    color_scheme='greens'
)

# Chart displays automatically in Jupyter
chart
```

### Data Format Conversion

```python
from latent_calendar.html import wide_to_long_format, dataframe_to_long_format

# Convert single row (168 columns) to long format
df_long = wide_to_long_format(
    df_states.loc['ca'],
    monday_start=True
)

print(df_long)
# Output:
#    day_of_week  hour  value
# 0            0     0     45
# 1            0     1     32
# ...
# 167          6    23     67

# Convert multiple rows for faceting
df_long_multi = dataframe_to_long_format(
    df_states.head(3),  # Multiple states
    group_col="state"
)

print(df_long_multi)
# Output:
#     state  day_of_week  hour  value
# 0      ca            0     0   45.0
# 1      ca            0     1   32.0
# ...
# 503    tx            6    23   28.0
```

## Color Schemes

The module supports all Altair/Vega color schemes. Default is `'greens'`.

**Sequential (single-hue):**
- `'greens'`, `'blues'`, `'reds'`, `'purples'`, `'greys'`, `'oranges'`

**Perceptual (multi-hue):**
- `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, `'cividis'`, `'turbo'`

**Diverging:**
- `'redblue'`, `'redgrey'`, `'blueorange'`, `'purpleorange'`

**Categorical:**
- `'category10'`, `'category20'`, `'tableau10'`, `'tableau20'`

See [Altair Color Schemes](https://vega.github.io/vega/docs/schemes/) for the complete list.

## Interactivity

By default, charts include:

- **Zoom/Pan**: Mouse wheel to zoom, click and drag to pan
- **Tooltips**: Hover over cells to see exact values
- **Reset**: Double-click to reset view

Disable interactivity:
```python
chart = create_calendar_chart(data, interactive=False)
```

## Customization Tips

### Compact Charts (No Values)

For smaller file sizes and cleaner appearance:
```python
chart = create_calendar_chart(
    data,
    show_values=False,  # Hide numeric labels
    width=350,          # Smaller dimensions
    height=250
)
html = chart.to_html()
```

### Sunday Week Start

To start weeks on Sunday instead of Monday:
```python
chart = create_calendar_chart(data, monday_start=False)
```

### Custom Embed Options

Control Altair rendering behavior:
```python
chart = create_calendar_chart(data)
html = chart.to_html(embed_options={
    'mode': 'vega-lite',
    'renderer': 'svg',  # Use SVG instead of Canvas
    'actions': False    # Hide action menu
})
```

## Technical Details

### Data Flow

**Single Calendar:**
```
Input: pd.Series with 168 values
  ↓
wide_to_long_format()
  ↓
DataFrame with (day_of_week, hour, value) columns
  ↓
create_calendar_chart()
  ↓
Altair Chart object
  ↓
chart.to_html() or chart.save()
  ↓
Standalone HTML document
```

**Faceted Calendars:**
```
Input: pd.DataFrame with multiple rows × 168 columns
  ↓
create_calendar_chart()  [auto-converts to long format internally]
  ↓
Altair Chart object
  ↓
chart.facet(column='group:N')
  ↓
Faceted Altair Chart object
  ↓
chart.to_html() or chart.save()
  ↓
Standalone HTML document

Note: Multi-row wide DataFrames are automatically converted to long format.
The index name is used as the group column (or "group" if unnamed).
```

### Chart Structure

The generated Altair chart uses:

- **Mark Type**: `rect` (rectangles for heatmap cells)
- **Encoding**:
  - X-axis: Hour of day (0-23)
  - Y-axis: Day of week (0-6)
  - Color: Event count/value
  - Tooltip: Day name, hour, value
- **Scale**: Sequential color scale based on data range

### HTML Output Format

The HTML includes:

- Complete `<!DOCTYPE html>` document
- Embedded Vega-Lite specification
- Bundled Vega/Vega-Lite/Vega-Embed JavaScript libraries
- All styles and scripts for standalone viewing

File size varies by chart complexity:
- Simple chart with `show_values=False`: ~50 KB
- Chart with values and large dataset: ~80-120 KB

## See Also

- [Folium Integration](../examples/folium-integration.md) - Use calendars in maps
- [`.cal` Accessor](./extensions.md) - Data aggregation and preparation
- [Altair Documentation](https://altair-viz.github.io/) - Chart customization
- [UFO Sightings Example](../examples/datasets/ufo-sightings.md) - Example dataset
