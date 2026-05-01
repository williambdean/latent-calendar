---
comments: true
---

# Generation Process

The `LatentCalendar` model is a generative model — after fitting, it can be used to
produce synthetic calendar data that reflects the patterns it discovered. This is useful
for simulation, testing, and understanding what the model has learned.

## Fitting a Model

Start with some calendar data in wide format and fit the model:

```python
from latent_calendar.generate import wide_format_dataframe
from latent_calendar import LatentCalendar

df = wide_format_dataframe(n_rows=50, rate=2.0, random_state=0)

model = LatentCalendar(n_components=5, random_state=0)
model.fit(df)
```

## Creating a Sampler

Once the model is fitted, create a sampler from it:

```python
sampler = model.create_sampler(random_state=42)
```

The sampler draws component mixture weights from the population-level Dirichlet prior
learned during fitting, then samples events from those weights.

## Sampling a Single User

Use `sample_events(n)` to generate `n` events for a single user:

```python
df_weights, df_events = sampler.sample_events(n=10)

# df_weights: (1, n_components) — the user's mixture over patterns
# df_events:  (1, 168)          — event counts across the weekly time slots
print(df_events.sum(axis=1))  # sums to 10
```

## Sampling Multiple Users

Pass a list to `sample()` to generate events for multiple users at once,
where each element is the number of events for that user:

```python
df_weights, df_events = sampler.sample(n_samples=[10, 5, 20])

# df_weights: (3, n_components)
# df_events:  (3, 168)
print(df_events.sum(axis=1))  # [10, 5, 20]
```

## Convenience Function

A convenience function is available if you don't need to reuse the sampler:

```python
from latent_calendar.generate import sample_from_latent_calendar

df_weights, df_events = sample_from_latent_calendar(
    model, n_samples=[10, 5, 20], random_state=42
)
```

## Visualising the Results

The returned event count DataFrame is in the same wide format as the training data,
so it can be passed directly to the `.cal` accessor for plotting:

```python
df_events.cal.plot_by_row()
```
