---
comments: true
---

# UFO Sightings

```python
import matplotlib.pyplot as plt

from latent_calendar.datasets import load_ufo_sightings

df = load_ufo_sightings()
```

Each row of the dataset is a UFO sighting somewhere around the world.

```text
            Date_time                  city state/province country  ...                                        description date_documented    latitude   longitude
0 1949-10-10 20:30:00            san marcos             tx      us  ...  This event took place in early fall around 194...       4/27/2004  29.8830556  -97.941111
1 1949-10-10 21:00:00          lackland afb             tx     NaN  ...  1949 Lackland AFB&#44 TX.  Lights racing acros...      12/16/2005    29.38421  -98.581082
2 1955-10-10 17:00:00  chester (uk/england)            NaN      gb  ...  Green/Orange circular disc over Chester&#44 En...       1/21/2008        53.2   -2.916667
3 1956-10-10 21:00:00                  edna             tx      us  ...  My older brother and twin sister were leaving ...       1/17/2004  28.9783333  -96.645833
4 1960-10-10 20:00:00               kaneohe             hi      us  ...  AS a Marine 1st Lt. flying an FJ4B fighter/att...       1/22/2004  21.4180556 -157.803611

[5 rows x 11 columns]
```

```python
df["year"] = df["Date_time"].dt.year

df_wide = df.cal.aggregate_events("year", "Date_time")

df_5_year = df_wide.tail(5)
```

which has the data in the wide format where each row is a weekly calendar.

```text
day_of_week  0                                                                                  1     ...  5     6
hour        0  1  2  3  4  5  6  7   8   9   10  11   12  13  14   15  16 17 18 19 20 21 22 23 0  1   ... 22 23 0  1  2  3  4  5  6  7  8  9   10  11  12  13  14   15 16 17 18 19 20 21 22 23
Country                                                                                               ...
Australia    0  0  0  0  0  0  0  0   0  27   6  19   20   7   2  105  27  0  0  0  0  0  0  0  0  0  ...  0  0  0  0  0  0  0  0  0  0  0  0  11  37  20   0  16    0  0  0  0  0  0  0  0  0
Austria      0  0  0  0  0  0  0  0   9  55  81   0   18   0   0   33   0  0  0  0  0  0  0  0  0  0  ...  0  0  0  0  0  0  0  0  0  0  0  0   0   0   0   0  37    0  0  0  0  0  0  0  0  0
Bahrain      0  0  0  0  0  0  0  0   0   0   1   6    0  13   0    0   0  0  0  0  0  0  0  0  0  0  ...  0  0  0  0  0  0  0  0  0  0  0  0   0   0   0   0   0    0  0  0  0  0  0  0  0  0
Belgium      0  0  0  0  0  0  0  0  15  15  42  26  109  54  38   17  26  0  0  0  0  0  0  0  0  0  ...  0  0  0  0  0  0  0  0  0  0  0  0  18  20  28  41   0  131  0  0  0  0  0  0  0  0
Bermuda      0  0  0  0  0  0  0  0   0   0   0   0    0   0   0    0   0  0  0  0  0  0  0  0  0  0  ...  0  0  0  0  0  0  0  0  0  0  0  0   0   0   0   0   0    0  0  0  0  0  0  0  0  0

[5 rows x 168 columns]
```

We can make use the [`cal` attribute](./../../modules/extensions.md) further to plot an aggregate of the data and plots of the data by day of week and hour of day.

```python

fig, axes = plt.subplots(ncols=3)

df_5_year.sum().cal.plot_row(ax=axes[0])

axes[0].set(
    title="Weekly UFO sightings",
)

for aggregation, ax in zip(["dow", "hour"], axes.ravel()[1:]):
    (
        df_5_year
        .cal.sum_over_vocab(aggregation=aggregation)
        .cal.divide_by_sum()
        .mul(100)
        .T.plot(ax=ax)
    )
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.1)

axes[1].set_xticks(axes[0].get_xticks())

axes[1].set(
    title="Chance of UFO sighting by day of week",
    xlabel="Day of week",
    ylabel="Chance of UFO sighting (%)",
)
axes[2].set(
    title="Chance of UFO sighting by hour of day",
    xlabel="Hour of day",
    ylabel="",
)
fig.suptitle("UFO sightings over 5 years")
plt.show()
```

![UFO Sightings](./../../images/weekly-ufo-sightings.png)
