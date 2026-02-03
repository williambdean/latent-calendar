---
comments: true
---
# Scikit-Learn Compatibility

The `LatentCalendar` class is `scikit-learn` compatible making it very easy to integrate into your existing machine learning pipelines.

Here are a few examples but feel free to get creative and combine them all together!

## Pipelines

`LatentCalendar` can be used in `scikit-learn` pipelines. The `tranform` method will be used based on the `TransformerMixin` base of `LatentDirichletAllocation`.


```python
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture

from latent_calendar import LatentCalendar


def create_hard_clustering_pipeline(n_components: int) -> Pipeline:
    gaussian_components = n_components ** 2 - 1
    return Pipeline(
        [
            ("latent_calendar", LatentCalendar(n_components=n_components)),
            ("gaussian_mixture", GaussianMixture(n_components=gaussian_components)),
        ]
    )

pipeline = create_hard_clustering_pipeline(n_components=3)
pipeline.fit(df_wide)

# Hard cluster labels
pipline.predict(df_wide)
```

## ColumnTransformer

Similar with `Pipeline`s, the `transform` method will be used.

```python
from sklearn.compose import ColumnTransformer

vocab_columns = df_wide.columns.tolist()

df_wide["total_events"] = df_wide.sum(axis=1)

transformer = ColumnTransformer(
    [
        ("latent_calendar", LatentCalendar(n_components=3), vocab_columns),
    ], remainder="passthrough"
)

transformer = transformer.fit(df_wide)
```

## Other Transformers

```python
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class RemoveLowVolumeTimeSlots(BaseEstimator, TransformerMixin):
    def __init__(self, model: LatentCalendar, min_count: int):
        self.model = model
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y=None) -> "RemoveLowVolumeTimeSlots":
        self.original_columns = X.columns

        column_counts = X.sum()
        self.columns_to_keep = column_counts[column_counts > self.min_count].index

        self.model.fit(X.loc[:, self.columns_to_keep])

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.model.predict(X.loc[:, self.columns_to_keep])

    def predict(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_pred = self.model.predict(X.loc[:, self.columns_to_keep])
        return X_pred.reindex(self.original_columns, fill_value=0, axis=1)


model = LatentCalendar(n_components=3)
transformer = RemoveLowVolumeTimeSlots(model=model, min_count=10)
transformer.fit(df_wide)
```
